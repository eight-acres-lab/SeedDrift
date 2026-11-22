"""The stages of package W2 on a small synthetic source run: rel_w2_v0 (ci/stages/rel/w2_v0.py) takes a passed run's
training arrays and homolog rows by their recorded checksums, decides G1 and G2 on the training genomes, leaves the
sealed test split unopened, and writes a table set the kind gates accept; rel_w2_census (ci/stages/rel/w2_census.py)
turns the counts into the report and its page. The fixture run is the one of test_rel_w2_tables.py with the groups and
taxids of the tiny corpus fixture."""
import json
import os

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w2_tables as T
from ci import manifest as M
from ci.gates import GateFailure
from ci.hashing import sha256_file
from ci.stages.common import Stage
from ci.stages.rel import common, gates as rel_gates, w2_census, w2_v0
from seeddrift.train.rel import handoff
from seeddrift.train.rel.evalh import score as SC
from seeddrift.train.rel.tables import v0_build as V
from seeddrift.train.rel.tables import v0_io as IO

# group: (taxid, source kind) of the tiny corpus fixture (tests/ci/fixtures/tiny/sources.tsv)
TINY = {"bacteria": (227377, "raw_prokaryote"), "mobile.plasmid": (227377, "plasmid"), "archaea": (190192, "raw_prokaryote"),
        "eukaryota.fungi": (4829, "annotated"), "eukaryota.invertebrates": (6052, "annotated"), "eukaryota.mammals": (9258, "annotated"),
        "eukaryota.other_vertebrates": (7757, "annotated"), "eukaryota.plants_algae": (2769, "annotated"), "eukaryota.protists": (2880, "annotated"),
        "viruses.eukaryotic": (10377, "annotated"), "viruses.giant": (10280, "annotated"), "viruses.phage": (10702, "annotated")}
SEALED = ["bacteria_GCF_000000001.1", "archaea_GCF_000000002.1"]
LEFT_OUT = "plasmid_GCF_000000001.1"                                 # a training plasmid of a sealed genome's assembly
GENOMES = ([("%s_%s%d" % (g.replace(".", "_"), s[:2], k), g, s) for g in sorted(TINY) for s, n in (("train", 2), ("heldout", 1)) for k in range(n)]
           + [(SEALED[0], "bacteria", "test"), (SEALED[1], "archaea", "test"), (LEFT_OUT, "mobile.plasmid", "train")])
SIZES = dict(train=25, heldout=12, test=2)


def make_source(run):
    """A passed selfsup run as stage rel_w2_v0 reads it: the arrays, their manifests, the stage records, the record."""
    metas, truth = T.make_run(run, genomes=GENOMES, lengths=[20, 24, 30, 33, 36, 40, 44, 50, 140, 270], taxids={g: v[0] for g, v in TINY.items()})
    ip = os.path.join(run, "data", "train_data", "index.json")
    tman = M.build("train-data", "train_data", [dict(id=m["id"], files=[M.file_entry(os.path.join(run, "data", "train_data", m["id"] + ".npz"), run)]) for m in metas])
    with open(ip) as f:
        index = json.load(f)
    with open(ip, "w") as f:
        json.dump(dict(index, sha256=M.write(tman, os.path.join(run, "manifests")), dataset_id=M.dataset_id(tman)), f)
    pman = M.build("pairs", "pairs", [dict(id=m["id"], files=[M.file_entry(os.path.join(run, "data", "pairs", m["id"] + ".npz"), run)]) for m in metas],
                   meta=dict(settings=dict(genomes=T.KEEP)))
    with open(os.path.join(run, "data", "pairs", "manifest.json"), "w") as f:
        json.dump(dict(sha256=M.write(pman, os.path.join(run, "manifests")), dataset_id=M.dataset_id(pman), **pman), f)
    cman = M.build("fixture", "corpus", [dict(id=m["id"], taxid=m["taxid"], source_kind=TINY[m["group"]][1], proteins=m["genes"], accession="", paired_accession="")
                                         for m in metas])
    os.makedirs(os.path.join(run, "data", "corpus"))
    with open(os.path.join(run, "data", "corpus", "manifest.json"), "w") as f:
        json.dump(dict(sha256=M.write(cman, os.path.join(run, "manifests")), dataset_id=M.dataset_id(cman), **cman), f)
    os.makedirs(os.path.join(run, "stages"))
    for stage, outs in (("train_data", [("train_data_index", "data/train_data/index.json")]), ("corpus", [("corpus_manifest", "data/corpus/manifest.json")]),
                        ("pairs", [("pairs_manifest", "data/pairs/manifest.json"), ("pairs_genomes", "data/pairs/genomes.json")])):
        with open(os.path.join(run, "stages", stage + ".json"), "w") as f:
            json.dump(dict(name=stage, status="passed", outputs=[dict(name=n, path=p, sha256=sha256_file(os.path.join(run, p), use_sidecar=False)) for n, p in outs]), f)
    with open(os.path.join(run, "record.json"), "w") as f:
        json.dump(dict(run_id=os.path.basename(run), status="passed", ci_version="0.3", git=dict(sha="1" * 40), pipeline="selfsup"), f)
    return metas, truth


def patch(mp, tmp):
    """This machine's sealed sources and the lineage check without the server's extra exclusion list (its taxids are
    not in the fixture taxonomy)."""
    spec = os.path.join(str(tmp), "sealed-test.json")
    with open(spec, "w") as f:
        json.dump(dict(extra_exclusions=[]), f)
    real = common.check_lineage
    mp.setattr(common, "sealed_set", lambda cfg, spec=None: RF.sealed(test_ids=SEALED, chromosomes=("NC_000913.3",)))
    mp.setattr(common, "check_lineage", lambda entries, cfg, tax=None, spec_path=None, **kw: real(entries, cfg, tax, spec, **kw))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w2")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas, truth = make_source(src)
    run_dir, cfg = RF.rel_run(tmp, stages=("w2_v0", "w2_census"), rel_w2_v0=dict(source=src, expect=SIZES, rows=16, train_per_group=6, heldout_per_group=2,
                                                                                  shard_tables=24, processes=2))
    with pytest.MonkeyPatch.context() as mp:
        patch(mp, tmp)
        with Stage(run_dir, "rel_w2_v0") as st:                          # the plain Stage: no machine check here
            doc = w2_v0.build(st, cfg, run_dir)
    return dict(tmp=tmp, src=src, metas=metas, truth=truth, run_dir=run_dir, cfg=cfg, doc=doc)


def _gate(run_dir, name):
    with open(os.path.join(run_dir, "gates", name + ".json")) as f:
        return json.load(f)


def test_stage_writes_a_table_set_of_the_open_splits(built):
    run_dir, doc = built["run_dir"], built["doc"]
    assert handoff.tableset_problems(doc, run_dir) == [] and doc["kind"] == "real" and doc["producer"]["stage"] == "w2_v0"
    assert doc["parts"]["train"]["tables"] == 72 and doc["parts"]["heldout"]["tables"] == 24 and doc["tables"] == 96
    assert [x["tables"] for x in doc["parts"]["train"]["shards"]] == [24, 24, 24]
    assert sorted({f["role"] for f in doc["files"]}) == ["index", "meta", "shard"]
    roles = {os.path.basename(f["path"]).split(".")[1]: f["role"] for f in doc["files"] if f["path"].endswith(".npy")}
    assert all(roles[k] == "shard" for k in ("tok", "col", "kind", "n")) and all(roles[k] == "meta" for k in ("genome", "identity", "group", "split", "tier", "gene"))
    with open(os.path.join(run_dir, w2_v0.OUT, "index.json")) as f:
        index = json.load(f)
    split = {m["id"]: m["split"] for m in built["metas"]}
    listed = [g["id"] for g in index["genomes"]]
    assert len(index["groups"]) == 12 and len(listed) == 36 and all(split[g] in ("train", "heldout") for g in listed)
    text = json.dumps(doc) + json.dumps(index) + open(os.path.join(run_dir, w2_v0.OUT, "counts.json")).read()
    assert not any(g in text for g in SEALED) and LEFT_OUT not in text       # sealed genomes and the left-out plasmid are named nowhere
    assert doc["left_out"] == [dict(group="mobile.plasmid", split="train", reasons=["the assembly of a sealed test genome"])]
    with open(os.path.join(run_dir, "stages", "rel_w2_v0.json")) as f:
        rep = json.load(f)
    assert rep["status"] == "passed" and rep["left_out"][0]["id"] == LEFT_OUT and rep["training_gates"] == dict(G1="pass", G2="pass", genomes=24)
    assert rep["split_sizes"] == SIZES and {o["name"] for o in rep["outputs"]} == {"tableset", "v0_index", "v0_counts"}
    for part, want in (("train", "train"), ("heldout", "heldout")):
        for x in doc["parts"][part]["shards"]:
            g = np.load(os.path.join(run_dir, w2_v0.OUT, part, x["name"] + ".genome.npy"))
            assert {split[listed[i]] for i in g[:, 0]} == {want} and {split[listed[i]] for i in g[:, 1:].ravel() if i >= 0} <= {"train"}
            assert (np.bincount(np.load(os.path.join(run_dir, w2_v0.OUT, part, x["name"] + ".group.npy")), minlength=12) == x["tables"] // 12).all()
    assert sorted(g["id"] for g in doc["genomes"]) == sorted(set(listed) & {g["id"] for g in doc["genomes"]}) and len(doc["genomes"]) > 24
    assert w2_v0.summary(run_dir)["json"]["tables"] == dict(train=72, heldout=24)
    assert doc["group_names"] == index["groups"] and doc["sealed_check"]["passed"] is True


def test_gates_pass_and_the_kind_gates_accept_the_table_set(built, monkeypatch):
    run_dir, cfg = built["run_dir"], built["cfg"]
    patch(monkeypatch, built["tmp"])
    with Stage(run_dir, "gates_rel_w2_v0") as st:
        w2_v0.gate(st, cfg, run_dir)
    assert [_gate(run_dir, g)["status"] for g in ("G1_rel_w2_v0", "G2_rel_w2_v0", "G4_rel_w2_v0")] == ["pass"] * 3
    assert _gate(run_dir, "G1_rel_w2_v0")["evidence"]["counts"] == {g: 2 for g in TINY}      # the training genomes alone
    assert _gate(run_dir, "G2_rel_w2_v0")["evidence"]["checked"] == 24
    with open(os.path.join(run_dir, "gates", "rel_w2_v0.ok"), "w") as f:
        json.dump(dict(phase="rel_w2_v0", passed=True), f)
    with Stage(run_dir, "rel_w2_census", requires=("pre", "rel_w2_v0")) as st:
        rep = w2_census.build(st, cfg, run_dir)
    with Stage(run_dir, "gates_rel_w2_census") as st:
        w2_census.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w2_census")["status"] == "pass"
    head = rep["headline"]
    st = rep["strata"]
    for part, n in (("train", 72), ("heldout", 24)):
        assert st[part]["tables"] == n and sum(x["tables"] for x in st[part]["fine"].values()) == n and "coarse" not in st[part]
        assert set(st[part]["per_group"]) == set(TINY) and 0 <= st[part]["high"]["from_0.98"]["share"] <= 1
        assert list(st[part]["fine"]) == list(SC.FINE) and list(head[part]["nearest_fine"]) == list(SC.FINE)
    with open(os.path.join(run_dir, w2_v0.OUT, "heldout", "s0000.identity.npy"), "rb") as f:
        ident = np.load(f)
    near = SC.nearest_identity(ident)
    assert st["heldout"]["high"]["from_0.98"]["tables"] == int((near >= 0.98 - 1e-6).sum()) and st["heldout"]["with_other_rows"] == int((~np.isnan(near)).sum())
    assert set(head["codons_per_group"]) == set(TINY) and all(v > 0 for v in head["codons_per_group"].values())
    assert head["train"]["tables"] == 72 and 0 < head["train"]["filled"]["not_target"] <= 1
    assert 0 <= head["train"]["random_rows_filled"]["not_target"] <= head["train"]["filled"]["not_target"]
    assert head["heldout"]["nearest_row_on_target"]["quantiles"]["q50"] is not None
    page = open(os.path.join(run_dir, w2_census.OUT, "report.md")).read()
    assert "## Table set `heldout`" in page and "viruses.giant" in page and not any(g in page for g in SEALED)
    assert "the strata of the evaluator" in page and "nearest_identity=0.9-1.0" in page and "at least 0.98 identical" in page
    cap = rep["cap"]                                                     # genes at the cap, against the homolog files themselves
    assert cap["cap"] == T.KEEP and cap["training_genomes"] == 24 and abs(cap["random_fill_ceiling"] - T.KEEP / 23) < 1e-12
    want = {}
    for m in built["metas"]:
        if m["split"] in ("train", "heldout") and m["id"] != LEFT_OUT:
            rows = np.diff(np.load(os.path.join(built["src"], "data", "pairs", m["id"] + ".npz"))["off"].astype(np.int64))
            d = want.setdefault(m["split"], [0, 0])
            d[0] += len(rows)
            d[1] += int((rows >= T.KEEP).sum())
    for split, (genes, at_cap) in want.items():
        assert cap["per_split"][split]["all"]["genes"] == genes and cap["per_split"][split]["all"]["genes_at_cap"] == at_cap
    assert all(at_cap > 0 for _, at_cap in want.values()) and head["train"]["genes_at_cap"]["all"] == cap["per_split"]["train"]["all"]["share_at_cap"]
    assert "reach the cap of stage pairs" in page and "measures the cap, not biology" in page and "counted over all three splits" in page
    for part in ("train", "heldout"):
        per = head[part]["tables_without_other_rows_per_group"]
        assert set(per) == set(TINY) and all(0 <= v <= 1 for v in per.values())
    assert w2_census.summary(run_dir)["json"] == head
    with open(os.path.join(run_dir, "gates", "rel_w2_census.ok"), "w") as f:
        json.dump(dict(phase="rel_w2_census", passed=True), f)
    with Stage(run_dir, "gates_rel_tables") as st:                           # the kind's gates: hand-off and the sealed check
        rel_gates.kind_phase(st, cfg, run_dir, ["w2_v0", "w2_census"])
    assert _gate(run_dir, "G4_rel_handoff")["status"] == "pass" and _gate(run_dir, "G3_rel")["status"] == "pass"


def test_round_trip_through_the_evaluator_reader_and_layout(built):
    """The shards read back through the manifest by the evaluator's reader (evalh/shards.py) are the files the build
    wrote, in the order of the tables; the evaluator's layout (evalh/layout.py place) puts every codon of every row in
    the column of the examined codon the simulation aligned it to, and nothing anywhere else."""
    from seeddrift.train.rel import api
    from seeddrift.train.rel.evalh import layout, shards
    run_dir, doc, truth, src = built["run_dir"], built["doc"], built["truth"], built["src"]
    sets = shards.by_split(run_dir, [doc])
    with open(os.path.join(run_dir, w2_v0.OUT, "index.json")) as f:
        listed = [g["id"] for g in json.load(f)["genomes"]]
    arrays = {}

    def gene(gid, j):
        if gid not in arrays:
            arrays[gid] = np.load(os.path.join(src, "data", "train_data", gid + ".npz"))
        z = arrays[gid]
        return z["codons"][z["codon_off"][j]:z["codon_off"][j + 1]]
    fields = sorted(IO.fields(16))
    for split, part in ((0, "train"), (1, "heldout")):
        ts = sets[split]
        assert ts.folder == os.path.join(w2_v0.OUT, part) and ts.tables == doc["parts"][part]["tables"]
        got = ts.load(0, ts.tables, fields)                                  # every file checked against the manifest
        direct = {f: np.concatenate([np.load(os.path.join(run_dir, w2_v0.OUT, part, "%s.%s.npy" % (x["name"], f))) for x in doc["parts"][part]["shards"]])
                  for f in fields}
        assert all(np.array_equal(got[f], direct[f], equal_nan=(f == "identity")) for f in fields)
        grid = layout.place(got["tok"], got["n"], got["col"], api.GAP)
        want = np.full(grid.shape, api.GAP, dtype=grid.dtype)
        for t in range(ts.tables):
            gid = listed[got["genome"][t, 0]]
            for b, (_, s0, s1) in enumerate(V.SLICES):
                i = int(got["gene"][t, b])
                if i < 0:
                    continue
                own = gene(gid, i)[:s1 - s0]
                want[t, 0, s0:s0 + len(own)] = own
                for q in range(1, 16):
                    if got["genome"][t, q] < 0 or (gid, i, listed[got["genome"][t, q]]) not in truth:
                        continue
                    pid = listed[got["genome"][t, q]]
                    j, pairs = truth[(gid, i, pid)]
                    other = gene(pid, j)
                    ok = {y: x for y, x in pairs.items() if x < len(own) and y < len(other)}
                    if not ok:
                        continue
                    first = min(ok)
                    for y, x in ok.items():
                        if y < first + (s1 - s0):                            # the stretch is cut at the block length
                            want[t, q, s0 + x] = other[y]
        assert np.array_equal(grid, want)


def test_census_of_another_run(built, monkeypatch, tmp_path):
    """The census alone, from the table set of a passed run (rel_w2_census.tables_from), as the census-only profile runs it."""
    import shutil
    patch(monkeypatch, built["tmp"])
    tables_run = str(tmp_path / "ci-root" / "runs" / "20261122T000001-tables")
    shutil.copytree(built["run_dir"], tables_run)
    with open(os.path.join(tables_run, "stages", "rel_w2_v0.json")) as f:
        stage = json.load(f)
    with open(os.path.join(tables_run, "record.json"), "w") as f:
        json.dump(dict(run_id=os.path.basename(tables_run), status="passed", stages=[stage], pipeline="rel_tables"), f)
    run_dir, cfg = RF.rel_run(tmp_path, stages=("w2_census",), run_id="20261122T000002-census", rel_w2_census=dict(tables_from=tables_run))
    with Stage(run_dir, "rel_w2_census") as st:
        rep = w2_census.build(st, cfg, run_dir)
    with Stage(run_dir, "gates_rel_w2_census") as st:
        w2_census.gate(st, cfg, run_dir)
    assert _gate(run_dir, "G4_rel_w2_census")["status"] == "pass" and rep["tableset"]["run_id"] == os.path.basename(built["run_dir"])
    assert rep["strata"]["heldout"]["tables"] == 24 and sum(x["tables"] for x in rep["strata"]["train"]["fine"].values()) == 72
    with Stage(run_dir, "gates_rel_tables") as st:                           # no table set written here: the kind gates pass
        rel_gates.kind_phase(st, cfg, run_dir, ["w2_census"])
    assert _gate(run_dir, "G3_rel")["status"] == "pass" and _gate(run_dir, "G4_rel_handoff")["status"] == "pass"
    with open(os.path.join(tables_run, "record.json"), "w") as f:
        json.dump(dict(run_id="x", status="failed", stages=[stage]), f)
    with pytest.raises(SystemExit, match="not passed"):
        w2_census.tables(run_dir, cfg)


def test_gate_fails_on_a_held_out_genome_among_the_other_rows(built, monkeypatch, tmp_path):
    import shutil
    run_dir, cfg = built["run_dir"], built["cfg"]
    patch(monkeypatch, built["tmp"])
    copy = str(tmp_path / "ci-root" / "runs" / os.path.basename(run_dir))
    shutil.copytree(run_dir, copy)
    with open(os.path.join(copy, w2_v0.OUT, "index.json")) as f:
        listed = json.load(f)["genomes"]
    p = os.path.join(copy, w2_v0.OUT, "train", "s0000.genome.npy")
    g = np.load(p)
    t = int(np.flatnonzero(g[:, 1] >= 0)[0])
    g[t, 1] = [i for i, x in enumerate(listed) if x["split"] == "heldout"][0]
    np.save(p, g)
    with pytest.raises(GateFailure):
        with Stage(copy, "gates_rel_w2_v0") as st:
            w2_v0.gate(st, dict(cfg, run_dir=copy), copy)
    assert _gate(copy, "G4_rel_w2_v0")["status"] == "fail" and _gate(copy, "G1_rel_w2_v0")["status"] == "pass"


def test_source_checks_fail(built, monkeypatch, tmp_path):
    import shutil
    patch(monkeypatch, built["tmp"])
    cfg = built["cfg"]
    assert w2_v0.source(built["src"])["provenance"]["run_id"] == "20261120T000000-src"
    with pytest.raises(SystemExit, match="unknown keys"):
        w2_v0.settings(dict(rel_w2_v0=dict(source="x", colour="blue")))
    with pytest.raises(SystemExit, match="split sizes"):
        w2_v0.open_genomes(w2_v0.source(built["src"]), dict(w2_v0.settings(cfg), expect=dict(train=711)), RF.sealed(test_ids=SEALED))
    with pytest.raises(SystemExit, match="at most"):
        w2_v0.open_genomes(w2_v0.source(built["src"]), dict(w2_v0.settings(cfg), max_left_out=0), RF.sealed(test_ids=SEALED))
    src = str(tmp_path / "src")
    shutil.copytree(built["src"], src)
    one = os.path.join(src, "data", "pairs", GENOMES[0][0] + ".npz")
    z = dict(np.load(one))
    z["qs"] = z["qs"] + 1                                                # a homolog file that is not the recorded one
    np.savez(one, **z)
    run_dir, cfg2 = RF.rel_run(tmp_path, stages=("w2_v0",), rel_w2_v0=dict(cfg["rel_w2_v0"], source=src, processes=1))
    with pytest.raises(SystemExit, match="differ from its manifests"):
        with Stage(run_dir, "rel_w2_v0") as st:
            w2_v0.build(st, cfg2, run_dir)
    assert not os.path.exists(os.path.join(run_dir, w2_v0.OUT, "tableset.json"))
    with open(os.path.join(src, "record.json"), "w") as f:
        json.dump(dict(run_id="x", status="failed"), f)
    with pytest.raises(SystemExit, match="not passed"):
        w2_v0.source(src)
    idx = os.path.join(built["src"], "data", "train_data", "index.json")
    assert V.sha256_of(idx) == sha256_file(idx, use_sidecar=False)
