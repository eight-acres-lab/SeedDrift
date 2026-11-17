"""G4: the comparators stand next to the model, on the same genes, and the released screen is recomputed in the run.

check(report, tier, skips, tag_commit) on data/eval/eval_report.json:
  - at least one scenario; every scenario has exactly one row of each method in METHODS;
  - every row names the gene set it was scored on (genes_sha256: ids and proteins in order) and it is the scenario's;
  - the model row is computed (a run that cannot score the model says nothing about it);
  - every other row is computed or "not_applicable: <reason>"; a computed row has its metrics;
  - released_v0.1.0 is computed in this run from the release tag (its provenance names the tag and the tag's commit,
    and the commit must be the one the tag points to), except in tier tiny, whose profile may skip it
    (`comparators: {released_v0.1.0: skip}`): GitHub runners have no reference data for the released screen.
"""
from __future__ import annotations

from ci.gates import GateResult
from ci.hashing import sha256_json

METHODS = ("model", "alien_index", "best_hit", "composition", "counts", "released_v0.1.0")
RELEASED = "released_v0.1.0"
TAG = "v0.1.0"


def check(report: dict, tier: str, skips=(), tag_commit: str = "") -> GateResult:
    problems = []
    scen = report.get("scenarios") or []
    if not scen:
        problems.append(dict(error="the report has no scenarios"))
    for sc in scen:
        name = sc.get("name", "?")
        methods = [r.get("method") for r in sc.get("rows", [])]
        if sorted(methods) != sorted(METHODS):
            problems.append(dict(scenario=name, error="rows %s, expected one each of %s" % (methods, list(METHODS))))
        gs = sc.get("genes_sha256")
        if not gs:
            problems.append(dict(scenario=name, error="the scenario names no gene set"))
        for r in sc.get("rows", []):
            m, st = r.get("method"), str(r.get("status", ""))
            if r.get("genes_sha256") != gs:
                problems.append(dict(scenario=name, method=m, error="scored on another gene set"))
            if st == "computed":
                if r.get("auroc") is None or r.get("recall_at_lowest_fpr") is None:
                    problems.append(dict(scenario=name, method=m, error="computed without metrics"))
            if m == "model" and st != "computed":
                problems.append(dict(scenario=name, method=m, error="the model is not scored: %s" % st))
            elif m == RELEASED:
                if st == "computed":
                    prov = r.get("provenance") or {}
                    if prov.get("tag") != TAG or not tag_commit or prov.get("commit") != tag_commit:
                        problems.append(dict(scenario=name, method=m, error="not recomputed from tag %s (provenance %s, tag commit %s)" % (TAG, prov, tag_commit or "unknown")))
                elif not (st.startswith("skipped") and tier == "tiny" and RELEASED in skips):
                    problems.append(dict(scenario=name, method=m, error="the released screen must be recomputed in every tier but tiny (status %s)" % st))
            elif m in METHODS and st != "computed" and not st.startswith("not_applicable"):
                problems.append(dict(scenario=name, method=m, error="status %r" % st))
    ev = dict(scenarios=len(scen), problems=problems[:50], tier=tier, skips=sorted(skips))
    return GateResult("G4", "fail" if problems else "pass", ev, sha256_json([report.get("scenarios"), tier, sorted(skips), tag_commit]))
