"""The model index of a run (handoff.models_problems): a trained variant lists its checkpoints (final, best or ema);
a probe or preflight-only run trains nothing and lists none (round 1's probes, stage w6_fit preflight_only)."""
from seeddrift.train.rel import handoff


def _doc(checkpoints, **flags):
    v = dict(name="arm-s1", status="ok", variant_file="configs/rel/variants/x.yaml", variant_sha256="0" * 64, api_version=1,
             info={}, checkpoints=checkpoints)
    return dict(format="rel-models", format_version=1, tablesets=[dict(run_id="r", name="t", sha256="0" * 64)], variants=[v], **flags)


def _errors(doc):
    return [p.get("error", "") for p in handoff.models_problems(doc)]


def test_a_trained_variant_needs_its_checkpoints():
    assert any("checkpoints are final, best or ema" in e for e in _errors(_doc([])))
    assert not any("checkpoint" in e for e in _errors(_doc([dict(which="final", path="final.pt", sha256="0" * 64)])))


def test_a_preflight_only_or_probe_run_lists_no_checkpoint():
    for flags in (dict(preflight_only=True), dict(probe=True)):
        assert not any("checkpoint" in e for e in _errors(_doc([], **flags)))
        assert any("lists no checkpoint" in e for e in _errors(_doc([dict(which="final", path="final.pt", sha256="0" * 64)], **flags)))
