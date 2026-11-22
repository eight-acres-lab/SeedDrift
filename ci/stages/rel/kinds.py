"""The pipeline kinds of the relational-model line (configs/rel/kinds.yaml; field meanings are documented there).

  load()           the registry, validated (cached per path)
  problems(doc)    what is wrong with a registry (empty when it is valid)
  is_kind(name)    whether a pipeline name is one of these kinds
  stages_of(cfg)   the stages a rel profile runs (rel.stages): known, of the profile's kind, distinct, not empty
  pipeline(cfg)    the run definition ci/__main__.py uses: target, required gates, resume phases
The existing pipelines (ci/__main__.py PIPELINES) are not changed by anything here.
"""
from __future__ import annotations

import os
import re

import yaml

from ci import ROOT

KINDS_FILE = os.path.join(ROOT, "configs", "rel", "kinds.yaml")
# the gates of phase pre, decided for every pipeline but select (ci/stages/gates.py phase_pre)
PRE_GATES = ("G6", "G7", "G7_prefix", "G7_resources", "G8", "G1_spec", "G2_static", "G3_static")
# names taken by the existing pipelines (rules, stage reports, data folders); never a stage of the new line
RESERVED = ("fit", "pairs", "select", "eval", "train", "search")
OLD_PIPELINES = ("train", "select", "search", "selfsup", "codon", "register", "audit", "family_a", "family_a_eval")
KIND_RX = re.compile(r"^rel_[a-z][a-z0-9]*$")
STAGE_RX = re.compile(r"^(c0|w[0-9]+)_[a-z][a-z0-9_]*$")
GATE_RX = re.compile(r"^G[0-9]+_rel(_[a-z0-9_]+)?$")
STAGE_KEYS = {"package", "rule", "output", "module", "gates", "side", "produces", "resumable", "tolerance", "stub"}
_CACHE = {}


def problems(doc) -> list:
    out = []
    if not isinstance(doc, dict) or doc.get("version") != 1 or not isinstance(doc.get("kinds"), dict) or not doc["kinds"]:
        return [dict(error="the registry needs version: 1 and a mapping kinds")]
    kg = doc.get("kind_gates") or []
    if not kg or any(not GATE_RX.match(str(g)) for g in kg):
        out.append(dict(error="kind_gates must name gates G<n>_rel...", kind_gates=kg))
    rules, outputs, gates = {}, {}, {g: "kind" for g in kg}
    for kind, kd in doc["kinds"].items():
        if not KIND_RX.match(kind) or kind in OLD_PIPELINES:
            out.append(dict(kind=kind, error="a kind is rel_<letters and digits> and no existing pipeline"))
        if not isinstance(kd, dict) or kd.get("lock") not in ("gpu", "select") or kd.get("target") != "rel_all" or not kd.get("stages"):
            out.append(dict(kind=kind, error="a kind has lock gpu or select, target rel_all and stages"))
            continue
        for name, s in kd["stages"].items():
            where = dict(kind=kind, stage=name)
            if not STAGE_RX.match(name):
                out.append(dict(where, error="a stage is <package>_<what> in lower case"))
                continue
            if not isinstance(s, dict) or set(s) - STAGE_KEYS:
                out.append(dict(where, error="unknown stage fields", fields=sorted(set(s) - STAGE_KEYS) if isinstance(s, dict) else None))
                continue
            pkg = name.split("_", 1)[0]
            if str(s.get("package", "")).lower() != pkg:
                out.append(dict(where, error="package %r does not match the stage prefix %s" % (s.get("package"), pkg)))
            if s.get("rule") != "rel_" + name:
                out.append(dict(where, error="the rule of stage %s is rel_%s" % (name, name)))
            if not str(s.get("output", "")).startswith("data/rel_%s/" % pkg) or ".." in str(s.get("output", "")):
                out.append(dict(where, error="the output lies under data/rel_%s/" % pkg))
            if not re.fullmatch(r"ci\.stages\.rel\.%s(_[a-z0-9_]+)?" % pkg, str(s.get("module", ""))):
                out.append(dict(where, error="the module is ci.stages.rel.%s_<what>" % pkg))
            if s.get("side") not in ("data", "train", "eval"):
                out.append(dict(where, error="side is data, train or eval"))
            if s.get("produces") not in (None, "tableset", "models"):
                out.append(dict(where, error="produces is tableset or models"))
            for flag in ("resumable", "tolerance", "stub"):
                if flag in s and not isinstance(s[flag], bool):
                    out.append(dict(where, error="%s is true or false" % flag))
            sg = s.get("gates") or []
            if not sg or any(not GATE_RX.match(str(g)) for g in sg):
                out.append(dict(where, error="a stage decides at least one gate G<n>_rel_..."))
            for g in sg:
                if g in gates:
                    out.append(dict(where, gate=g, error="gate also decided by %s" % gates[g]))
                gates[g] = name
            for table, key in ((rules, s.get("rule")), (outputs, s.get("output"))):
                if key in table:
                    out.append(dict(where, error="%s also belongs to stage %s" % (key, table[key])))
                table[key] = name
    all_stages = [s for kd in doc["kinds"].values() if isinstance(kd, dict) for s in (kd.get("stages") or {})]
    if len(all_stages) != len(set(all_stages)):
        out.append(dict(error="a stage name is used by two kinds"))
    return out


def load(path: str = KINDS_FILE) -> dict:
    if path not in _CACHE:
        with open(path) as f:
            doc = yaml.safe_load(f)
        bad = problems(doc)
        if bad:
            raise SystemExit("configs/rel/kinds.yaml is not valid: %s" % bad[:5])
        _CACHE[path] = doc
    return _CACHE[path]


def is_kind(name, path: str = KINDS_FILE) -> bool:
    return isinstance(name, str) and name.startswith("rel_") and name in load(path)["kinds"]


def all_stages(path: str = KINDS_FILE) -> dict:
    """{stage: its fields plus kind}"""
    return {s: dict(d, kind=k) for k, kd in load(path)["kinds"].items() for s, d in kd["stages"].items()}


def stage(name, path: str = KINDS_FILE) -> dict:
    st = all_stages(path)
    if name not in st:
        raise SystemExit("unknown stage %r of the relational-model line (configs/rel/kinds.yaml)" % name)
    return st[name]


def stages_of(cfg, path: str = KINDS_FILE) -> list:
    kind = cfg.get("pipeline")
    if not is_kind(kind, path):
        raise SystemExit("pipeline %r is not a kind of configs/rel/kinds.yaml" % kind)
    on = (cfg.get("rel") or {}).get("stages")
    if not isinstance(on, list) or not on or len(set(on)) != len(on):
        raise SystemExit("a %s profile lists the stages it runs, each once, in rel.stages" % kind)
    mine = load(path)["kinds"][kind]["stages"]
    other = [s for s in on if s not in mine]
    if other:
        raise SystemExit("rel.stages %s are not stages of kind %s (its stages: %s)" % (other, kind, ", ".join(mine)))
    return list(on)


def pipeline(cfg, path: str = KINDS_FILE) -> dict:
    """target, required gates and resume phases of a rel run. A resume re-decides phase pre and every phase whose
    sentinel exists (those are the phases Snakemake will not run again)."""
    doc = load(path)
    kind = cfg["pipeline"]
    on = stages_of(cfg, path)
    st = doc["kinds"][kind]["stages"]
    required = list(PRE_GATES) + [g for s in on for g in st[s]["gates"]] + list(doc["kind_gates"])
    resume = [("pre", None)] + [("rel_" + s, "gates/rel_%s.ok" % s) for s in on] + [(kind, "gates/%s.ok" % kind)]
    return dict(target=doc["kinds"][kind]["target"], required=tuple(required), resume=tuple(resume), kind=kind, stages=tuple(on),
                lock=doc["kinds"][kind]["lock"])
