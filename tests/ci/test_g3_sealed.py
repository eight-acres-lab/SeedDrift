import json
import os

from ci.gates import g3_sealed


def _sealed(tmp_path):
    p = tmp_path / "sealed.tsv"
    p.write_text("accession\tgenus\ttier\nNC_005945.1\tBacillus\tsealed\nNC_002516.2\tPseudomonas\tsealed\n")
    return str(p)


def test_static_clean_files_pass(tmp_path):
    cfg = tmp_path / "config.resolved.yaml"
    cfg.write_text("corpus_spec: configs/corpus/smoke.yaml\naccession: NC_0059451\n")     # a longer number is another accession
    assert g3_sealed.static([str(cfg)], "smoke", _sealed(tmp_path)).status == "pass"


def test_static_sealed_accession_fails(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text("sources:\n  - {id: x, ena_sequences: [NC_005945]}\n")               # without its version too
    r = g3_sealed.static([str(spec)], "smoke", _sealed(tmp_path))
    assert r.status == "fail" and r.evidence["found"]


def test_freeze_tier_fails_until_it_has_rules(tmp_path):
    assert g3_sealed.static([], "freeze", _sealed(tmp_path)).status == "fail"


def test_built_evaluation_genome_under_sealed_genus_fails(tmp_path, tax):
    run = tmp_path / "run"
    (run / "data" / "eval_inputs").mkdir(parents=True)
    (run / "data" / "eval_inputs" / "manifest.json").write_text(json.dumps({"entries": [{"id": "e1"}]}))
    sealed = _sealed(tmp_path)
    assert g3_sealed.built(str(run), tax, {"e1": [562]}, "smoke", sealed_path=sealed).status == "pass"
    r = g3_sealed.built(str(run), tax, {"e1": [1423]}, "smoke", sealed_path=sealed)           # Bacillus subtilis
    assert r.status == "fail" and "e1" in r.evidence["under_sealed_genus"]
    # the insect genus Bacillus (a homonym) is not the sealed bacterial genus
    assert g3_sealed.built(str(run), tax, {"e1": [999901]}, "smoke", sealed_path=sealed).status == "pass"


def test_built_sealed_accession_in_a_manifest_fails(tmp_path, tax):
    run = tmp_path / "run"
    (run / "data" / "fetch").mkdir(parents=True)
    (run / "data" / "fetch" / "manifest.json").write_text(json.dumps({"entries": [{"provenance": [{"url": "x/NC_002516.2"}]}]}))
    r = g3_sealed.built(str(run), tax, {}, "smoke", sealed_path=_sealed(tmp_path))
    assert r.status == "fail" and r.evidence["accessions_found"] == {"fetch/manifest.json": ["NC_002516.2"]}


def test_real_sealed_list_loads():
    rows = g3_sealed.load_sealed()
    assert len(rows) >= 17 and all(a.startswith("NC_") for a, _ in rows)
