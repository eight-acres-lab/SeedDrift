"""The pipeline kinds of the relational-model line (configs/rel/kinds.yaml, ci/stages/rel/kinds.py): the registry is
valid, every stage's rule is where the registry says and writes what it says, no rule or output name is taken twice,
and ci/__main__.py derives a kind's target, gates and resume phases from the stages a profile runs."""
import json
import os
import re

import pytest

from conftest import ROOT
from ci.stages.rel import kinds

RULE_RX = re.compile(r"^rule (\w+):", re.MULTILINE)


def _rules():
    """{rule name: [(file, output line)]} over ci/Snakefile and every ci/rules/*.smk."""
    out = {}
    files = [os.path.join(ROOT, "ci", "Snakefile")] + sorted(os.path.join(ROOT, "ci", "rules", f) for f in os.listdir(os.path.join(ROOT, "ci", "rules")))
    for p in files:
        text = open(p).read()
        for m in RULE_RX.finditer(text):
            body = text[m.end():].split("\nrule ", 1)[0]
            o = re.search(r"^\s+output:\s*(.+)$", body, re.MULTILINE)
            out.setdefault(m.group(1), []).append((os.path.relpath(p, ROOT), o.group(1).strip() if o else ""))
    return out


def test_registry_is_valid():
    with open(kinds.KINDS_FILE) as f:
        import yaml
        assert kinds.problems(yaml.safe_load(f)) == []
    assert set(kinds.load()["kinds"]) == {"rel_tables", "rel_traineval", "rel_readout", "rel_corpus", "rel_events"}


def test_bad_registries_fail():
    good = kinds.load()
    bad = json.loads(json.dumps(good))
    bad["kinds"]["rel_tables"]["stages"]["c0_synth"]["output"] = "data/select/x.json"          # outside the package's folder
    bad["kinds"]["rel_corpus"]["stages"]["w13_census"]["rule"] = "pairs"                      # an old rule name
    bad["kinds"]["rel_events"]["lock"] = "both"
    bad["kinds"]["train"] = bad["kinds"]["rel_events"]                                       # an existing pipeline
    probs = kinds.problems(bad)
    assert len(probs) >= 4
    dup = json.loads(json.dumps(good))
    dup["kinds"]["rel_events"]["stages"]["w15_registry"]["gates"] = ["G4_rel_c0_synth"]      # a gate decided twice
    assert any("gate also decided" in p["error"] for p in kinds.problems(dup))


def test_every_stage_rule_is_in_its_package_file_with_its_output():
    rules = _rules()
    for name, d in kinds.all_stages().items():
        where = rules.get(d["rule"]) or []
        assert len(where) == 1, (name, where)
        f, out = where[0]
        assert f == "ci/rules/rel_%s.smk" % d["package"].lower(), (name, f)
        assert out.strip('"').strip("'") == d["output"], (name, out)


def test_rule_names_and_outputs_are_unique_and_prefixed():
    rules = _rules()
    assert all(len(v) == 1 for v in rules.values()), {k: v for k, v in rules.items() if len(v) > 1}
    outs = [o for v in rules.values() for _, o in v if o]
    assert len(outs) == len(set(outs))
    for name, v in rules.items():
        f = v[0][0]
        if f.startswith("ci/rules/rel_"):
            pkg = f[len("ci/rules/rel_"):-len(".smk")]
            assert name.startswith("rel_%s_" % pkg) and name not in kinds.RESERVED, (f, name)


def test_pipeline_from_the_stages_a_profile_runs():
    from ci.__main__ import PIPELINES, pipeline
    p = pipeline(dict(pipeline="rel_tables", rel=dict(stages=["c0_synth"])))
    assert p["target"] == "rel_all" and p["lock"] == "select"
    assert p["required"] == kinds.PRE_GATES + ("G4_rel_c0_synth", "G3_rel", "G4_rel_handoff")
    assert p["resume"] == (("pre", None), ("rel_c0_synth", "gates/rel_c0_synth.ok"), ("rel_tables", "gates/rel_tables.ok"))
    t = pipeline(dict(pipeline="rel_traineval", rel=dict(stages=["w4_train", "w5_eval"])))
    assert t["lock"] == "gpu" and "G4_rel_w4_train" in t["required"] and "G4_rel_w5_report" not in t["required"]
    assert "rel_tables" not in PIPELINES                       # the existing pipelines are unchanged


@pytest.mark.parametrize("cfg", [dict(pipeline="rel_tables", rel=dict(stages=[])), dict(pipeline="rel_tables", rel=dict(stages=["w4_train"])),
                                 dict(pipeline="rel_tables", rel=dict(stages=["c0_synth", "c0_synth"])), dict(pipeline="rel_tables"),
                                 dict(pipeline="rel_nothing", rel=dict(stages=["c0_synth"]))])
def test_bad_stage_lists_fail(cfg):
    from ci.__main__ import pipeline
    with pytest.raises(SystemExit):
        pipeline(cfg)


def test_gate_phases_accept_the_rel_kinds():
    from ci.stages.gates import pipeline_of
    assert pipeline_of(dict(pipeline="rel_corpus")) == "rel_corpus"
    with pytest.raises(SystemExit):
        pipeline_of(dict(pipeline="rel_unknown"))


def test_run_record_schema_takes_rel_kinds_only_by_pattern_fails_otherwise():
    import jsonschema
    with open(os.path.join(ROOT, "ci", "schemas", "run_record.schema.json")) as f:
        prop = json.load(f)["properties"]["pipeline"]
    for ok in ("train", "family_a_eval", "rel_tables", "rel_x2"):
        jsonschema.validate(ok, prop)
    for bad in ("rel_", "rel-tables", "relx", "Rel_tables"):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, prop)


def test_a_train_side_stage_reading_ruler_sets_fails():
    bad = json.loads(json.dumps(kinds.load()))
    bad["kinds"]["rel_traineval"]["stages"]["w6_fit"]["reads"] = ["real", "ruler"]
    assert any("never reads ruler" in p["error"] for p in kinds.problems(bad))
    bad["kinds"]["rel_traineval"]["stages"]["w6_fit"]["reads"] = ["rulers"]
    assert any("reads is a list" in p["error"] for p in kinds.problems(bad))
    assert kinds.reads("w6_eval") == ("real", "ruler") and kinds.reads("w6_fit") == ("real",)
