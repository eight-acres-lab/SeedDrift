"""Benchmark v2 M0, slim (ci/stages/audit.py): the numpy fused baselines, the controls, the columns, and the pipeline
end to end on synthetic GenBank records with gate G3_audit and G4_audit."""
import csv
import json
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ci.stages import audit as AU  # noqa: E402
from ci.stages import protocol as PR  # noqa: E402


def test_l1_logistic_is_sparse_and_finds_the_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 6))
    y = (X[:, 2] + 0.3 * rng.normal(size=2000) > 0).astype(float)
    mu, sd, w = AU.l1_logistic_fit(X, y, lam=0.02)
    assert w[3] > 1.0 and np.sum(np.abs(w[1:]) > 1e-6) <= 3
    s = AU.l1_logistic_score((mu, sd, w), X)
    assert PR.auroc(s[y == 1], s[y == 0]) > 0.95


def test_gbm_fits_an_interaction_a_linear_model_misses():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(3000, 3))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    m = AU.gbm_fit(X, y, rounds=100, depth=2, lr=0.2, min_leaf=20)
    s = AU.gbm_score(m, X)
    lin = AU.l1_logistic_score(AU.l1_logistic_fit(X, y, lam=0.0), X)
    assert PR.auroc(s[y == 1], s[y == 0]) > 0.9 > 0.6 > PR.auroc(lin[y == 1], lin[y == 0])
    X2 = X.copy()
    X2[:5, 0] = np.nan                                   # missing values go to their own bin
    assert np.isfinite(AU.gbm_score(m, X2)).all()


def test_gbm_splits_a_rare_value_of_a_binary_feature():
    rng = np.random.default_rng(6)
    x = (rng.random(4000) < 0.02).astype(float)            # 2 % ones: quantile edges alone would merge them with the zeros
    y = np.where(x > 0, rng.random(4000) < 0.9, rng.random(4000) < 0.1).astype(float)
    assert len(AU.bin_edges(x, 32)) == 1 and AU.bin_edges(x, 32)[0] == 0.5
    m = AU.gbm_fit(x[:, None], y, rounds=50, depth=1, lr=0.3, min_leaf=20)
    s = AU.gbm_score(m, x[:, None])
    assert s[x > 0].min() > s[x == 0].max()                 # the rare value is split off
    assert abs(PR.auroc(s[y == 1], s[y == 0]) - PR.auroc(x[y == 1], x[y == 0])) < 1e-12


def _table(rng, flip_unit=None):
    table = {}
    for k, u in enumerate(["u1", "u1", "u2", "u3"]):
        n = 300
        y = (rng.random(n) < 0.2).astype(int)
        x = y * (-1.5 if u == flip_unit else 1.5) + rng.normal(size=n)
        table["c%d" % k] = dict(y=y, unit=u, cols=dict(x=x, z=rng.normal(size=n)))
    return table


def test_leave_unit_out_never_trains_on_the_held_out_unit():
    rng = np.random.default_rng(3)
    table = _table(rng, flip_unit="u3")                 # in u3 the signal points the other way
    feats = lambda t: np.c_[t["cols"]["x"], t["cols"]["z"]]
    sc = AU.leave_unit_out(table, ["u1", "u2", "u3"], feats, "l1", dict(lam=0.01))
    t3 = table["c3"]
    a3 = PR.auroc(sc["c3"][t3["y"] == 1], sc["c3"][t3["y"] == 0])
    assert a3 < 0.2                                     # a model that had seen u3 could not rank it this badly
    assert set(sc) == set(table)


def test_shuffle_control_comes_back_to_one_half():
    rng = np.random.default_rng(4)
    out = AU.shuffle_control(_table(rng), ["x", "z"], draws=20, seed=1)
    assert abs(out["x"] - 0.5) < 0.02 and abs(out["z"] - 0.5) < 0.02


def test_cai_is_highest_for_the_reference_usage():
    rng = np.random.default_rng(5)
    pref = rng.dirichlet(np.ones(64) * 0.3)
    other = rng.dirichlet(np.ones(64) * 3)
    counts = np.stack([rng.multinomial(300, pref) for _ in range(30)] + [rng.multinomial(300, other) for _ in range(10)]).astype(float)
    ref = np.zeros(40, bool)
    ref[:25] = True
    c = AU.cai(counts, ref)
    assert c[25:30].mean() > c[30:].mean() and np.all((c > 0) & (c <= 1))


def _record(path, acc, rng, n=180, islands=range(60, 85), trna=(10, 120)):
    """A circular GenBank record: n coding genes of 300 bp on the plus strand, an island of donor-like GC, ribosomal
    protein products for 25 genes, phage words on some island genes, two tRNA genes."""
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO
    host = np.array([0.3, 0.2, 0.2, 0.3])
    donor = np.array([0.15, 0.35, 0.35, 0.15])
    seq, feats, pos = [], [], 0
    stops = {"TAA", "TAG", "TGA"}
    for i in range(n):
        p = donor if i in islands else host
        cod = ["ATG"]
        while len(cod) < 99:
            c = "".join(rng.choice(list("ACGT"), 3, p=p))
            if c not in stops:
                cod.append(c)
        cod.append("TAA")
        s = "".join(cod)
        prod = "30S ribosomal protein S%d" % i if i < 25 else ("phage integrase" if i in islands and i % 3 == 0 else "hypothetical protein")
        q = dict(locus_tag=["%s_%04d" % (acc.replace(".", "_"), i)], product=[prod], transl_table=["11"], translation=[str(Seq(s).translate(table=11, to_stop=True))])
        feats.append(SeqFeature(FeatureLocation(pos, pos + 300, strand=1), type="CDS", qualifiers=q))
        seq.append(s)
        pos += 300
        seq.append("ACGTACGTAC")
        if i in trna:
            feats.append(SeqFeature(FeatureLocation(pos - 10, pos, strand=1), type="tRNA", qualifiers=dict(product=["tRNA-Ala"])))
        pos += 10
    rec = SeqRecord(Seq("".join(seq)), id=acc, name=acc.split(".")[0], description="synthetic", annotations=dict(molecule_type="DNA", topology="circular"))
    rec.annotations["data_file_division"] = "BCT"
    rec.features = [SeqFeature(FeatureLocation(0, pos, strand=1), type="source", qualifiers={"db_xref": ["taxon:562"]})] + feats
    SeqIO.write(rec, str(path), "genbank")


def _fixture(tmp_path):
    """An island run (inputs built by the island stage's own build_inputs, arrays, manifest, stage records, gate, the
    evaluator's dev.json) and a store (records, genes tables, gated labels) for two development chromosomes."""
    pytest.importorskip("Bio")
    from ci import manifest as M
    from ci.hashing import sha256_file
    from ci.stages import islands as I
    from ci.stages.common import read_fasta
    from ci.stages.search import gene_ids
    from seeddrift.inputs import from_genbank
    rng = np.random.default_rng(7)
    store, isl = tmp_path / "store", tmp_path / "runs" / "isl"
    for d in ("dev_genomes", "genes", "labels_panel"):
        (store / d).mkdir(parents=True)
    (isl / "gates").mkdir(parents=True)
    accs = [("NC_A1.1", "Escherichia"), ("NC_A2.1", "Salmonella")]
    genomes, files, dev_rows = [], [], []
    res = dict(per_chromosome={}, eligibility={}, units={})
    for acc, genus in accs:
        gb = store / "dev_genomes" / (acc + ".gb")
        _record(gb, acc, rng)
        g = from_genbank(gb.read_text())
        _, ren = I.clean_names(g.proteins)
        with open(store / "genes" / (acc + ".tsv"), "w") as f:
            f.write("gene_id\tcontig\tstart\tend\tstrand\n")
            for gid, new in ren.items():
                c, s, e, strand, _ = g.coords[gid]
                f.write("%s\t%s\t%d\t%d\t%s\n" % (new, c, s, e, strand))
        folder = isl / "data" / "islands" / "inputs" / acc
        I.build_inputs(str(store), acc, str(folder))
        names = gene_ids(str(folder))
        cds = read_fasta(str(folder / "cds.fna"))
        tok = []
        off = [0]
        for nme in names:
            s = cds["%s__%s" % (acc, nme)]
            t = [16 * "ACGT".index(s[j]) + 4 * "ACGT".index(s[j + 1]) + "ACGT".index(s[j + 2]) + 1 for j in range(0, len(s) - 2, 3)]
            tok += t
            off.append(len(tok))
        n = len(names)
        arr = isl / "data" / "islands" / "train_data" / (acc + ".npz")
        arr.parent.mkdir(parents=True, exist_ok=True)
        start = np.arange(n) * 310 + 1
        np.savez(arr, codons=np.array(tok, np.uint8), codon_off=np.array(off, np.int64), replicon=np.zeros(n, np.int64), start=start, end=start + 299,
                 strand=np.ones(n, np.int8), replicon_len=np.array([n * 310]), hit_off=np.zeros(n + 1, np.int64), hit_prot=np.zeros(0, np.int32),
                 hit_qs=np.zeros(0, np.int32), track=np.zeros(0, np.uint8), track_off=np.zeros(1, np.int64))
        files += [os.path.relpath(arr, isl), os.path.relpath(folder / "genes.tsv", isl), os.path.relpath(folder / "cds.fna", isl)]
        genomes.append(dict(id=acc, genus=genus, taxid=562, ranks=[1, 2, 3, 4, 5, 6], group="bacteria", path=os.path.relpath(arr, isl), folder=os.path.relpath(folder, isl)))
        with open(store / "labels_panel" / (acc + ".tsv"), "w") as f:
            f.write("gene_id\tcontig\tstart\tend\tstrand\tlabel\tregion\tproduct\tclass\tpublished\tf\n")
            for i, nme in enumerate(names):
                c = "island" if 60 <= i < 85 else "edge" if 55 <= i < 90 else "native"
                f.write("%s\t0\t%d\t%d\t+\t%s\t\t\t%s\t%s\t%s\n" % (nme, start[i], start[i] + 299, "island" if c == "island" else "backbone" if c == "native" else "none",
                                                                     c, "island" if c == "island" else "backbone", "0.2" if c == "island" else "1.0"))
        res["per_chromosome"][acc] = dict(v1=dict(gated=dict(auroc=0.5)))
        res["eligibility"][acc] = dict(dev_gene=True)
        res["units"][acc] = genus
        dev_rows.append((genus, acc, sha256_file(str(gb), use_sidecar=False)))
    man = M.build("islands", "islands", [dict(id=p, files=[M.file_entry(str(isl / p), str(isl))]) for p in files])
    msha = M.write(man, str(isl / "manifests"))
    (isl / "data" / "islands" / "manifest.json").write_text(json.dumps(dict(sha256=msha)))
    (isl / "data" / "islands" / "index.json").write_text(json.dumps(dict(genomes=genomes)))
    (isl / "data" / "islands" / "eval").mkdir()
    (isl / "data" / "islands" / "eval" / "dev.json").write_text(json.dumps(res))
    (isl / "config.resolved.yaml").write_text(yaml.safe_dump(dict(islands=dict(store=str(store)))))
    (isl / "stages").mkdir()
    for stage, name, p in (("islands_data", "islands_index", isl / "data" / "islands" / "index.json"), ("islands_report", "islands_eval", isl / "data" / "islands" / "eval" / "dev.json")):
        (isl / "stages" / (stage + ".json")).write_text(json.dumps(dict(status="passed", outputs=[dict(name=name, sha256=sha256_file(str(p), use_sidecar=False))])))
    (isl / "gates" / "G3_islands.json").write_text(json.dumps(dict(status="pass")))
    dev_tsv = tmp_path / "dev_genomes.tsv"
    dev_tsv.write_text("genus\taccession\torganism\tsha256\n" + "".join("%s\t%s\tx\t%s\n" % r for r in dev_rows))
    ref_tsv = tmp_path / "v2_reference_set.tsv"
    ref_tsv.write_text("accession\torganism\tgenus\tfamily\torder\tpart\tlevel\tstatus\tlevel_from\n"
                       + "".join("%s\tx\t%s\t\t\tdevelopment\tL0\tdevelopment\t\n" % (a, g) for a, g in accs)
                       + "NC_R99.1\tBurkholderia y\tBurkholderia\t\t\trest\tL1\tcandidate\t\n")
    return isl, store, dev_tsv, ref_tsv


def test_audit_end_to_end(tmp_path, monkeypatch):
    from ci.stages import common as CM
    from ci.stages import gates as GA
    isl, store, dev_tsv, ref_tsv = _fixture(tmp_path)
    redirect = {"bench/islands/results/dev_genomes.tsv": str(dev_tsv), "bench/hard/results/v2_reference_set.tsv": str(ref_tsv)}
    real_repo = CM.Stage.repo
    monkeypatch.setattr(CM.Stage, "repo", lambda self, rel: redirect.get(rel) or real_repo(self, rel))
    run = tmp_path / "runs" / "ours"
    (run / "gates").mkdir(parents=True)
    (run / "gates" / "pre.ok").write_text("{}")
    cfg = dict(pipeline="audit", seed=1, audit=dict(islands_from=str(isl), fused=dict(rounds=30, min_leaf=10), boot=50, controls=dict(draws=20, tolerance=0.05)))
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    AU.do_columns(str(run))
    with open(run / "data" / "audit" / "columns" / "scores_NC_A1.1.tsv") as f:
        head = f.readline().rstrip("\n").split("\t")
    assert {"tv_gc", "tv_cai", "tv_win_gc", "tv_mobile", "cmp_kmer"} <= set(head)
    with open(run / "data" / "audit" / "columns" / "meta_NC_A1.1.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows[10]["md_trna_dist"] in ("0.0", "1.0") and float(rows[50]["md_trna_dist"]) > 10
    AU.do_report(str(run))
    rep = json.loads((run / "data" / "audit" / "report.json").read_text())
    assert rep["scores"]["tv_gc"]["auroc"] > 0.9 and abs(rep["negative_control"]["tv_gc"] - 0.5) < 0.05
    assert rep["scores"]["tv_win_gc"]["auroc"] >= rep["edge_cell"]["tv_win_gc"]           # the label margin inflates windows
    assert rep["detector"]["core"]["auroc_within"] is not None and "fz_gbm" in rep["scores"]
    assert rep["detector"]["core"]["flag"]              # the fixture's islands sit at one place in both records: a planted shortcut
    st = type("St", (), dict(repo=staticmethod(lambda rel: redirect.get(rel) or (rel if os.path.isabs(rel) else os.path.join(ROOT, rel)))))
    GA.phase_audit(st, cfg, str(run))
    assert json.loads((run / "gates" / "G3_audit.json").read_text())["status"] == "pass"
    assert json.loads((run / "gates" / "G4_audit.json").read_text())["status"] == "pass"
    # a report without its shuffle control fails G4_audit
    rp = run / "data" / "audit" / "report.json"
    keep = rp.read_text()
    bad = json.loads(keep)
    bad["negative_control"] = {}
    rp.write_text(json.dumps(bad))
    with pytest.raises(Exception):
        GA.phase_audit(st, cfg, str(run))
    assert json.loads((run / "gates" / "G4_audit.json").read_text())["status"] == "fail"
    rp.write_text(keep)
    # a register accession of another part in an audit output fails G3_audit
    (run / "data" / "audit" / "note.txt").write_text("compare with NC_R99.1")
    with pytest.raises(Exception):
        GA.phase_audit(st, cfg, str(run))
    assert json.loads((run / "gates" / "G3_audit.json").read_text())["status"] == "fail"
