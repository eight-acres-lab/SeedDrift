"""Load the exact parameter file from a pinned, passed W5 training fit."""
import hashlib
import json
import re
from pathlib import Path


STAGE = "rel_w5_baselines"


def _output(run, outputs, name, expected):
    path = "data/rel_w5/baselines/%s.json" % name
    entries = [o for o in outputs if o["name"] == name]
    if len(entries) != 1 or entries[0]["path"] != path:
        raise ValueError("frozen fit: missing or ambiguous %s output" % name)
    data = (run / path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entries[0]["sha256"] or digest != expected:
        raise ValueError("frozen fit: %s sha256 differs from its record or pin" % name)
    return data


def load_frozen_fit(spec):
    """Return the fitted parameters, their original bytes and their checked provenance."""
    run = Path(spec["run"])
    record = json.loads((run / "record.json").read_bytes())
    commit = spec["commit"]
    if record["run_id"] != run.name or not re.fullmatch("[0-9a-f]{40}", commit) or record.get("git", {}).get("sha") != commit:
        raise ValueError("frozen fit: source run ID or Git commit differs from the pin")
    stages = [s for s in record["stages"] if s["name"] == STAGE]
    if record["status"] != "passed" or len(stages) != 1 or stages[0]["status"] != "passed":
        raise ValueError("frozen fit: the source run and its W5 fit stage must have passed")
    if record.get("gates", {}).get("G4_rel_w5_baselines") != "pass":
        raise ValueError("frozen fit: the source baseline gate must have passed")
    outputs = stages[0]["outputs"]
    results = json.loads(_output(run, outputs, "results", spec["results_sha256"]))
    raw = _output(run, outputs, "params", spec["params_sha256"])
    params = json.loads(raw)

    fit = results["fit"]
    source = results["tables_run"]
    sets = [s for s in source["tablesets"] if s["name"] == fit["set"]]
    if len(sets) != 1 or sets[0]["sha256"] != spec["train_tableset_sha256"]:
        raise ValueError("frozen fit: training table manifest differs from its pin")
    if results["stage"] != STAGE or fit["tables"] != spec["fit_tables"] or fit["seed"] != spec["fit_seed"]:
        raise ValueError("frozen fit: source stage, fit count or seed differs from the requested fit")
    if "source" in fit or stages[0].get("tables_fitted") != fit["tables"]:
        raise ValueError("frozen fit: the source stage did not perform the recorded training fit")
    if not 0 < fit["tables"] <= sets[0]["tables"] or fit["seed"] != results["seed"] + 1:
        raise ValueError("frozen fit: inconsistent source count or seed")
    held = results["heldout"]
    if (fit["set"], fit["folder"]) == (held["set"], held["folder"]) or fit["files_read"] < 1:
        raise ValueError("frozen fit: missing evidence of a separate training input")

    if not params or set(params) != set(fit["numbers"]):
        raise ValueError("frozen fit: parameter masks differ from the fit record")
    for mask, values in params.items():
        metadata = values["fit"]
        if (metadata["mask"], metadata["tables"], metadata["seed"]) != (mask, fit["tables"], fit["seed"]):
            raise ValueError("frozen fit: inconsistent training metadata for %s" % mask)
        if not values["prior"] or {k: v for k, v in values.items() if k != "prior"} != fit["numbers"][mask]:
            raise ValueError("frozen fit: fitted numbers differ from the source results for %s" % mask)

    provenance = dict(run_id=record["run_id"], commit=commit, tables_run_id=source["run_id"], set=fit["set"],
                      folder=fit["folder"], files_read=fit["files_read"], fit_tables=fit["tables"], fit_seed=fit["seed"],
                      train_tableset_sha256=sets[0]["sha256"], params_sha256=spec["params_sha256"],
                      results_sha256=spec["results_sha256"], original_layouts=results["layouts"], original_items=results["items"])
    return params, raw, provenance
