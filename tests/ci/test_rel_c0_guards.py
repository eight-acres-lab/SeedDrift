"""The guards of the relational-model line: the sealed check (seeddrift/train/rel/sealed.py and the lineage half in
ci/stages/rel/common.py), "this configuration belongs to this machine" (check_machine) and the hand-off files
(seeddrift/train/rel/handoff.py). Each passes a clean case and fails every case it exists for."""
import hashlib
import json
import os

import pytest

import rel_fixtures as RF
from conftest import ROOT
from ci.stages.rel import common
from seeddrift.train.rel import handoff, sealed as S

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


# --- sealed check -----------------------------------------------------------------------------------------------

def test_clean_ids_pass_the_sealed_check():
    s = RF.sealed(test_ids=["bacteria_GCF_000005845.2"], chromosomes=["NC_010410.1"])
    ev = S.check_ids(["bacteria_GCF_000006765.1", "archaea_GCA_000007185.1"], s, splits={"bacteria_GCF_000006765.1": "train", "archaea_GCA_000007185.1": "heldout"})
    assert ev["checked"] == 2


@pytest.mark.parametrize("gid,splits", [("bacteria_GCF_000005845.2", None),            # a sealed test genome
                                         ("plasmid_gca_000005845", None),               # its assembly, other prefix, case and no version
                                         ("x__nc_010410", None),                        # a sealed chromosome inside a composite name
                                         ("bacteria_GCF_000006765.1", {"bacteria_GCF_000006765.1": "test"}),
                                         ("bacteria_GCF_000006765.1", {"bacteria_GCF_000006765.1": "eval"}),
                                         ("bacteria_GCF_000006765.1", {})])             # split unknown
def test_sealed_check_fails(gid, splits):
    s = RF.sealed(test_ids=["bacteria_GCF_000005845.2"], chromosomes=["NC_010410.1"])
    with pytest.raises(S.SealedError):
        S.check_ids([gid], s, splits=splits)


def _ci_root(tmp_path, test=("g_test_GCF_000000003.1",), train=("g_train_GCF_000000001.1",), registry=None):
    root = tmp_path / "root"
    idx = root / "imports" / "runs" / "R" / "data" / "train_data" / "index.json"
    idx.parent.mkdir(parents=True)
    idx.write_text(json.dumps(dict(genomes=[dict(id=g, split="test") for g in test] + [dict(id=g, split="train") for g in train])))
    if registry is not None:
        (root / "host").mkdir()
        (root / "host" / "sealed_registry.json").write_text(json.dumps(registry))
    spec = dict(version=1, test_split=dict(run="R", path="data/train_data/index.json", sha256=hashlib.sha256(idx.read_bytes()).hexdigest(),
                                           search=["runs", "imports/runs"], split="test", expect=len(test)),
                registry="host/sealed_registry.json", chromosome_lists=["bench/hard/results/sealed.tsv"], extra_exclusions=[])
    sp = tmp_path / "sealed.json"
    sp.write_text(json.dumps(spec))
    return str(root), str(sp), spec


def test_sealed_set_loads_the_pinned_split_and_the_registry(tmp_path):
    reg = dict(corpus_test_splits=[dict(genomes=[dict(id="g_test_GCF_000000003.1", aliases=["GCA_000000003.2"])])],
               chromosomes=[dict(accession="NC_999999.1", aliases=["NC_999999"])])
    root, sp, _ = _ci_root(tmp_path, registry=reg)
    s = S.SealedSet.load(root, sp, ROOT)
    assert s.source["registry"] and s.source["test_genomes"] == 1
    assert S.check_ids(["g_train_GCF_000000001.1"], s)["checked"] == 1
    for bad in ("g_test_GCF_000000003.1", "GCA_000000003.2", "other_gcf_000000003", "nc_999999.4", "NC_010410"):
        with pytest.raises(S.SealedError):
            S.check_ids([bad], s)


def test_sealed_set_fails_on_a_wrong_index_count_or_registry(tmp_path):
    root, sp, spec = _ci_root(tmp_path)
    for change in (dict(sha256="0" * 64), dict(expect=2), dict(run="missing")):
        bad = dict(spec, test_split=dict(spec["test_split"], **change))
        with open(sp, "w") as f:
            json.dump(bad, f)
        with pytest.raises(S.SealedError):
            S.SealedSet.load(root, sp, ROOT)
    root2, sp2, _ = _ci_root(tmp_path / "b", registry=dict(corpus_test_splits=[dict(genomes=[dict(id="another")])]))
    with pytest.raises(S.SealedError):                                              # the registry disagrees with the pin
        S.SealedSet.load(root2, sp2, ROOT)


def test_lineage_check_passes_clean_genomes_and_fails_an_excluded_one(tax, tmp_path):
    spec = tmp_path / "sealed.json"
    spec.write_text(json.dumps(dict(extra_exclusions=[])))
    paths = dict(exclude_path=os.path.join(FIX, "mini_exclude.tsv"), panels_path=str(tmp_path / "panels.tsv"), catalogue_path=str(tmp_path / "none.tsv"))
    (tmp_path / "panels.tsv").write_text("assembly\tspecies_taxid\trole\tpanel_of\n")
    cfg = dict(taxdump=os.path.join(FIX, "mini_taxdump"))
    assert common.check_lineage([dict(id="aceto", taxid=481146)], cfg, tax, str(spec), **paths)["checked"] == 1
    with pytest.raises(SystemExit):
        common.check_lineage([dict(id="k12", taxid=83333)], cfg, tax, str(spec), **paths)            # under the development order
    with pytest.raises(SystemExit):
        common.check_lineage([dict(id="ghost", taxid=424242)], cfg, tax, str(spec), **paths)         # an unknown taxid fails closed


def test_the_repository_sealed_spec_names_existing_lists():
    with open(S.SPEC) as f:
        spec = json.load(f)
    assert spec["test_split"]["expect"] == 257 and len(spec["test_split"]["sha256"]) == 64
    for rel in spec["chromosome_lists"] + spec["extra_exclusions"]:
        assert os.path.exists(os.path.join(ROOT, rel)), rel


# --- this configuration belongs to this machine ------------------------------------------------------------------

def _cfg(tmp_path, machine="i1", **rel):
    root = common.machines()[machine]["ci_root"]
    return dict(rel=dict(dict(machine=machine), **rel), ci_root=root, run_dir=os.path.join(root, "runs", "x"))


def test_the_right_machine_passes(tmp_path):
    facts = common.check_machine(_cfg(tmp_path), gpus=RF.gpus("i1"))
    assert facts["machine"] == "i1" and facts["parallel"] == 1
    assert common.check_machine(_cfg(tmp_path, "i2"), gpus=RF.gpus("i2"))["machine"] == "i2"


@pytest.mark.parametrize("change", ["other_gpu", "two_gpus", "no_gpu", "ci_root", "outside", "parallel", "unknown"])
def test_another_machine_fails(tmp_path, change):
    cfg, gpus = _cfg(tmp_path), RF.gpus("i1")
    if change == "other_gpu":
        gpus = RF.gpus("i2")
    elif change == "two_gpus":
        gpus = gpus * 2
    elif change == "no_gpu":
        gpus = []
    elif change == "ci_root":
        cfg["ci_root"] = "/elsewhere"
    elif change == "outside":
        cfg["run_dir"] = "/tmp/run"
    elif change == "parallel":
        cfg = _cfg(tmp_path, "i2", parallel=2)
        gpus = RF.gpus("i2")
    elif change == "unknown":
        cfg["rel"]["machine"] = "i9"
    with pytest.raises(SystemExit):
        common.check_machine(cfg, gpus=gpus)


def test_rel_stage_refuses_another_machine_with_a_stage_report_fails(tmp_path):
    run_dir, _ = RF.rel_run(tmp_path)
    with pytest.raises(SystemExit):
        with common.RelStage(run_dir, "rel_c0_synth", gpus=RF.gpus("i1")):
            pass                                        # the scratch CI root is not machine i1's
    rep = json.load(open(os.path.join(run_dir, "stages", "rel_c0_synth.json")))
    assert rep["status"] == "failed"


# --- hand-off files ----------------------------------------------------------------------------------------------

def test_a_written_tableset_is_valid_and_any_change_fails(tmp_path):
    run_dir, _ = RF.rel_run(tmp_path)
    doc = RF.write_tableset(run_dir)
    assert handoff.tableset_problems(doc, run_dir) == []
    assert handoff.tableset_problems(dict(doc, kind="real"), run_dir) != []                       # digest no longer matches
    assert handoff.tableset_problems(handoff.finish(dict(doc, kind="real")), run_dir) != []       # a real set without genomes
    assert handoff.tableset_problems(handoff.finish(dict(doc, coverage=dict(note="x"))), run_dir) != []
    shard = os.path.join(run_dir, doc["files"][0]["path"])
    with open(shard, "ab") as f:
        f.write(b"!")
    assert any("content" in p["error"] for p in handoff.tableset_problems(doc, run_dir))


def test_model_index_rules_fail_on_missing_parts(tmp_path):
    ck = tmp_path / "final.bin"
    ck.write_bytes(b"w")
    base = dict(format="rel-models", format_version=1, tablesets=[dict(run_id="r", name="t", sha256="a" * 64)],
                producer=dict(stage="w4_train", package="W4", run_id="r", commit="0" * 40),
                variants=[dict(name="ref", variant_file="configs/rel/variants/w6-ref.yaml", variant_sha256="b" * 64, api_version=1, status="ok", info={},
                               checkpoints=[dict(which="final", budget=None, path="final.bin", sha256=handoff._file_sha(str(ck)), bytes=1)]),
                          dict(name="bad", variant_file="x", variant_sha256="c" * 64, api_version=1, status="failed", error="out of memory", info={}, checkpoints=[])])
    assert handoff.models_problems(handoff.finish(base), str(tmp_path)) == []
    v = [dict(base["variants"][0], checkpoints=[dict(base["variants"][0]["checkpoints"][0], which="latest")])]
    assert handoff.models_problems(handoff.finish(dict(base, variants=v)), str(tmp_path)) != []
    assert handoff.models_problems(handoff.finish(dict(base, tablesets=[])), str(tmp_path)) != []
    v2 = [dict(base["variants"][1], error="")]
    assert handoff.models_problems(handoff.finish(dict(base, variants=v2)), str(tmp_path)) != []


def test_from_run_takes_a_passed_runs_listed_file_and_fails_otherwise(tmp_path):
    run_dir, _ = RF.rel_run(tmp_path)
    doc = RF.write_tableset(run_dir)
    out = "data/rel_c0/synth/tableset.json"
    sha = handoff._file_sha(os.path.join(run_dir, out))
    rec = dict(run_id="r", status="passed", stages=[dict(name="rel_c0_synth", outputs=[dict(name="tableset", path=out, sha256=sha)])])
    with open(os.path.join(run_dir, "record.json"), "w") as f:
        json.dump(rec, f)
    assert handoff.from_run(run_dir, out, doc["sha256"])["name"] == doc["name"]
    with pytest.raises(handoff.HandoffError):
        handoff.from_run(run_dir, out, "f" * 64)
    with open(os.path.join(run_dir, "record.json"), "w") as f:
        json.dump(dict(rec, status="failed"), f)
    with pytest.raises(handoff.HandoffError):
        handoff.from_run(run_dir, out)
