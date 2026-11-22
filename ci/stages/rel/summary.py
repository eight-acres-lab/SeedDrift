"""The run-summary hook of the relational-model line (called by ci/__main__.py summary for pipelines rel_*).

hook(run_dir, record) -> (block for summary.json, lines for summary.md). For every stage the run lists in rel.stages
whose module defines summary(run_dir) -> dict(json=<small JSON value>, md=[lines]), that stage's part is added under
its name. A stage without the function adds its status only; a failing function is reported, never raised (the summary
is written in every case). The summary files are published (ci/publish.py: at most 200 KB together, no home paths, no
sealed accession), so a stage's part stays small: counts and headline numbers, not tables.
"""
from __future__ import annotations

import importlib
import json
import os

import yaml

from ci.stages.rel import kinds

PART_MAX = 20_000            # bytes of one stage's JSON part


def hook(run_dir, record):
    out, md = dict(kind=record.get("pipeline"), stages={}), []
    try:
        with open(os.path.join(run_dir, "config.resolved.yaml")) as f:
            cfg = yaml.safe_load(f)
        on = kinds.stages_of(cfg)
    except BaseException as err:  # noqa: BLE001 - the summary is written whatever happens here
        out["error"] = "no stage list: %s" % err
        return out, ["", "## Relational-model line", "", "summary hook: %s" % out["error"]]
    status = {s["name"]: s["status"] for s in record.get("stages") or []}
    md += ["", "## Relational-model line: %s" % record.get("pipeline"), "", "| stage | package | status |", "|---|---|---|"]
    parts = []
    for s in on:
        d = kinds.stage(s)
        part = dict(status=status.get("rel_" + s, "not run"), package=d["package"])
        md.append("| %s | %s | %s |" % (s, d["package"], part["status"]))
        if not d.get("stub") and part["status"] == "passed":
            try:
                fn = getattr(importlib.import_module(d["module"]), "summary", None)
                if fn is not None:
                    got = fn(run_dir) or {}
                    js = got.get("json")
                    if len(json.dumps(js, default=str)) > PART_MAX:
                        js = dict(error="the stage's summary part exceeds %d bytes" % PART_MAX)
                    part["summary"] = js
                    if got.get("md"):
                        parts += ["", "### %s" % s, ""] + [str(x) for x in got["md"]][:200]
            except BaseException as err:  # noqa: BLE001
                part["summary_error"] = "%s: %s" % (type(err).__name__, str(err)[:300])
        out["stages"][s] = part
    return out, md + parts
