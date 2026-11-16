"""Gates. Each module gN_*.py exposes check(ctx) -> GateResult. run() writes runs/<id>/gates/<gate>.json and returns
the result; a failing gate makes the calling stage exit non-zero. There is no force flag: the only override is a
waiver file in ci/waivers/ (gate, scope, reason, approved_by, date, expires), and an expired waiver fails.

Gates fail closed: an unknown taxid, an unknown group or an unreadable input is a failure, not a skip.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
from dataclasses import dataclass, field

import yaml

from ci import ROOT
from ci.record import atomic_write_json, now

WAIVERS = os.path.join(ROOT, "ci", "waivers")
WILDCARD_GATES = {"G5"}            # only these gates may be waived with scope "*"
MAX_DAYS = 90                      # a waiver lives at most this long


@dataclass
class GateResult:
    gate: str
    status: str                       # pass | fail | waived
    evidence: dict = field(default_factory=dict)
    inputs_hash: str = ""
    waiver: dict = None

    def as_dict(self):
        return dict(gate=self.gate, status=self.status, evidence=self.evidence, inputs_hash=self.inputs_hash, waiver=self.waiver,
                    waivers_matched=list(MATCHED), checked=now())


class GateFailure(RuntimeError):
    pass


def waivers(today=None, folder=WAIVERS):
    """Valid waivers; an expired or malformed one raises (it must be removed or renewed, not ignored)."""
    today = today or _dt.date.today()
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.yml"))):
        with open(p) as f:
            w = yaml.safe_load(f) or {}
        for k in ("gate", "scope", "reason", "approved_by", "date", "expires"):
            if not w.get(k):
                raise GateFailure("waiver %s lacks %s" % (p, k))
        exp = w["expires"] if isinstance(w["expires"], _dt.date) else _dt.date.fromisoformat(str(w["expires"]))
        start = w["date"] if isinstance(w["date"], _dt.date) else _dt.date.fromisoformat(str(w["date"]))
        if exp < today:
            raise GateFailure("waiver %s expired on %s" % (p, exp))
        if (exp - start).days > MAX_DAYS:
            raise GateFailure("waiver %s runs %d days; at most %d are allowed" % (p, (exp - start).days, MAX_DAYS))
        if w["scope"] == "*" and w["gate"] not in WILDCARD_GATES:
            raise GateFailure("waiver %s: scope '*' is not allowed for gate %s" % (p, w["gate"]))
        w["file"] = os.path.relpath(p, ROOT)
        out.append(w)
    return out


MATCHED = []                       # waivers matched in this process (recorded in the gate results and the run record)


def find_waiver(gate: str, scope: str, folder=WAIVERS):
    for w in waivers(folder=folder):
        if w["gate"] == gate and (w["scope"] == scope or (w["scope"] == "*" and gate in WILDCARD_GATES)):
            MATCHED.append(dict(gate=w["gate"], scope=scope, file=w["file"], expires=str(w["expires"])))
            return w
    return None


def run(result: GateResult, run_dir: str = "") -> GateResult:
    if run_dir:
        folder = os.path.join(run_dir, "gates")
        os.makedirs(folder, exist_ok=True)
        atomic_write_json(os.path.join(folder, result.gate + ".json"), result.as_dict())
    return result


def require(result: GateResult, run_dir: str = "") -> GateResult:
    run(result, run_dir)
    if result.status == "fail":
        raise GateFailure("%s failed: %s" % (result.gate, json.dumps(result.evidence, ensure_ascii=False)[:2000]))
    return result
