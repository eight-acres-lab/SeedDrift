"""The two files packages hand to each other (format version 1): the table-set manifest and the model index.
Standard library only, so the CI environment and the model's Python read them the same way.

Table-set manifest (`tableset.json`, written by a stage with `produces: tableset` in configs/rel/kinds.yaml):
  format "rel-tableset", format_version 1
  name           unique within the run (letters, digits, . _ -)
  kind           real, synthetic or ruler; training rounds accept real table sets only, the probe mode synthetic ones
  api_version    the version of the contract (seeddrift/train/rel/api.py) the shards follow
  table_version  the version string of the code that built the tables
  coverage       what the set covers and what it does not (task rules: every manifest says so): groups (list),
                 scenarios (list), not_covered (list), note
  tables         the number of tables
  files          [{path, sha256, bytes, role}] relative to the run directory; role shard (what models read), index,
                 meta (what models never read: genome ids, groups, splits, distance bins, identities, cell presence;
                 for samplers and evaluators only), truth (synthetic sets only: the correct sources, for tests only)
  genomes        [{id, taxid, split}] every genome that occurs in a row; [] for synthetic sets. The sealed check reads it.
  purpose        train, heldout, probe or ruler; a train set may hold genomes of the train split only (gate G3_rel)
  not_for_training  true for a set no training may read (W12's tables of the island set's development chromosomes,
                 whose rows of split island are the 24 development chromosomes, task breakdown 8.8; gate G3_rel allows
                 that split only here, after ci/stages/rel/common.py development_chromosomes); common.read_tableset
                 refuses such a set to every train-side stage
  sealed_check   real and ruler sets: the record of the passed sealed check of the stage that built them
                 (ci/stages/rel/common.py sealed_record): passed true, the ids checked, the sealed sources with the
                 registry. Every stage that reads a table set takes it through from_run, which refuses a set without it.
  producer       {stage, package, run_id, commit}
  sha256         sha256 of the canonical JSON of every other field (written last)
Model index (`models.json`, written by the training stage, read by evaluation and readouts):
  format "rel-models", format_version 1
  tablesets      [{run_id, name, sha256}] the table sets the models were trained on
  variants       [{name, variant_file, variant_sha256, api_version, status (ok or failed), error, info, checkpoints:
                 [{which (final, best or ema), budget (null or the budget point), path, sha256, bytes}]}]
  producer, sha256 as above
A package may add fields; it may not drop or change these without a new format version (a request to the coordinator).

  digest(doc)                         the sha256 of a manifest (its sha256 field left out)
  finish(doc)                         doc with its sha256 set
  tableset_problems(doc, run_dir)     what is wrong with a table-set manifest (and its files, given run_dir)
  models_problems(doc, run_dir)       what is wrong with a model index
  sealed_problems(doc)                what is wrong with the sealed record of a real or ruler table set
  from_run(run_dir, path, expect_sha256=None, kinds=None)
                                      a hand-off file of another run: the run passed, its stage record lists the file
                                      with this content, the manifest's digest is its sha256 and, when given, the one
                                      expected, a real or ruler table set carries a passed sealed record, and, when kinds
                                      is given, a table set is of one of those kinds; returns the parsed manifest
"""
from __future__ import annotations

import hashlib
import json
import os
import re

KINDS = ("real", "synthetic", "ruler")
PURPOSES = ("train", "heldout", "probe", "ruler")
ROLES = ("shard", "index", "meta", "truth")
NAME_RX = re.compile(r"^[A-Za-z0-9._-]+$")
SHA_RX = re.compile(r"^[0-9a-f]{64}$")


class HandoffError(ValueError):
    pass


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(doc) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in doc.items() if k != "sha256"})).hexdigest()


def finish(doc) -> dict:
    out = {k: v for k, v in doc.items() if k != "sha256"}
    out["sha256"] = digest(out)
    return out


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _files(doc, run_dir, roles):
    out = []
    files = doc.get("files")
    if not isinstance(files, list) or not files:
        return [dict(error="no files")]
    seen = set()
    for f in files:
        if not isinstance(f, dict) or not {"path", "sha256", "bytes", "role"} <= set(f):
            out.append(dict(file=f, error="a file entry has path, sha256, bytes and role"))
            continue
        p = str(f["path"])
        if p.startswith("/") or ".." in p.split("/") or p in seen:
            out.append(dict(file=p, error="paths are relative to the run directory, inside it, each once"))
        seen.add(p)
        if f["role"] not in roles or not SHA_RX.match(str(f["sha256"])):
            out.append(dict(file=p, error="role one of %s and a sha256" % ", ".join(roles)))
            continue
        if run_dir:
            full = os.path.join(run_dir, p)
            if not os.path.exists(full) or os.path.getsize(full) != f["bytes"] or _file_sha(full) != f["sha256"]:
                out.append(dict(file=p, error="missing or not the content the manifest names"))
    return out


def _common(doc, fmt):
    out = []
    if not isinstance(doc, dict) or doc.get("format") != fmt or doc.get("format_version") != 1:
        return [dict(error="not a %s manifest of format version 1" % fmt)]
    if doc.get("sha256") != digest(doc):
        out.append(dict(error="the sha256 field is not the digest of the manifest"))
    p = doc.get("producer")
    if not isinstance(p, dict) or not {"stage", "package", "run_id", "commit"} <= set(p):
        out.append(dict(error="producer names stage, package, run_id and commit"))
    return out


def tableset_problems(doc, run_dir="") -> list:
    out = _common(doc, "rel-tableset")
    if out and "format version" in out[0].get("error", ""):
        return out
    if not NAME_RX.match(str(doc.get("name", ""))):
        out.append(dict(error="the name is letters, digits, . _ -"))
    if doc.get("kind") not in KINDS:
        out.append(dict(error="kind is one of %s" % ", ".join(KINDS)))
    for k in ("api_version", "table_version"):
        if doc.get(k) in (None, ""):
            out.append(dict(error="%s is missing" % k))
    cov = doc.get("coverage")
    if not isinstance(cov, dict) or not all(isinstance(cov.get(k), list) for k in ("groups", "scenarios", "not_covered")) or not cov.get("note"):
        out.append(dict(error="coverage has the lists groups, scenarios and not_covered, and a note"))
    if not isinstance(doc.get("tables"), int) or doc["tables"] <= 0:
        out.append(dict(error="tables is a positive count"))
    roles = ROLES if doc.get("kind") == "synthetic" else tuple(r for r in ROLES if r != "truth")
    out += _files(doc, run_dir, roles)
    if not any(isinstance(f, dict) and f.get("role") == "shard" for f in doc.get("files") or []):
        out.append(dict(error="no shard"))
    g = doc.get("genomes")
    if not isinstance(g, list) or any(not isinstance(x, dict) or not x.get("id") for x in g):
        out.append(dict(error="genomes is a list of {id, taxid, split}"))
    elif doc.get("kind") == "synthetic" and g:
        out.append(dict(error="a synthetic table set has no genomes"))
    elif doc.get("kind") != "synthetic" and (not g or any("taxid" not in x or "split" not in x for x in g)):
        out.append(dict(error="a real or ruler table set lists every genome with its taxid and split"))
    if doc.get("purpose") is not None and doc["purpose"] not in PURPOSES:
        out.append(dict(error="purpose is one of %s" % ", ".join(PURPOSES)))
    if doc.get("not_for_training") not in (None, True, False) or (doc.get("not_for_training") and doc.get("purpose") == "train"):
        out.append(dict(error="not_for_training is true or false, and never on a set of purpose train"))
    if any(isinstance(x, dict) and x.get("split") == "island" for x in doc.get("genomes") or []) and not doc.get("not_for_training"):
        out.append(dict(error="genomes of split island only in a set marked not_for_training"))
    out += sealed_problems(doc)
    return out


def sealed_problems(doc) -> list:
    if not isinstance(doc, dict) or doc.get("kind") == "synthetic":
        return []
    rec = doc.get("sealed_check")
    if not isinstance(rec, dict) or rec.get("passed") is not True:
        return [dict(error="a real or ruler table set carries the record of a passed sealed check (sealed_check.passed)")]
    ids = rec.get("ids") or {}
    want = len({g.get("id") for g in doc.get("genomes") or [] if isinstance(g, dict)})
    if not isinstance(ids.get("distinct"), int) or ids["distinct"] < want or not ((ids.get("source") or {}).get("registry")):
        return [dict(error="the sealed record checked %s distinct ids against %s; it must cover at least the %d genomes listed, with the "
                     "machine's registry" % (ids.get("distinct"), (ids.get("source") or {}).get("kind"), want))]
    return []


def models_problems(doc, run_dir="") -> list:
    out = _common(doc, "rel-models")
    if out and "format version" in out[0].get("error", ""):
        return out
    ts = doc.get("tablesets")
    if not isinstance(ts, list) or not ts or any(not isinstance(t, dict) or not {"run_id", "name", "sha256"} <= set(t) for t in ts):
        out.append(dict(error="tablesets lists {run_id, name, sha256}"))
    vs = doc.get("variants")
    if not isinstance(vs, list) or not vs:
        return out + [dict(error="no variants")]
    names = [v.get("name") for v in vs if isinstance(v, dict)]
    if len(names) != len(vs) or len(set(names)) != len(names):
        out.append(dict(error="variants have distinct names"))
    for v in vs:
        if not isinstance(v, dict):
            continue
        if v.get("status") not in ("ok", "failed") or not {"variant_file", "variant_sha256", "api_version", "info", "checkpoints"} <= set(v):
            out.append(dict(variant=v.get("name"), error="a variant has status ok or failed, variant_file, variant_sha256, api_version, info, checkpoints"))
            continue
        if v["status"] == "failed":
            if not v.get("error"):
                out.append(dict(variant=v["name"], error="a failed variant says why"))
            continue
        cks = v["checkpoints"]
        if doc.get("preflight_only") or doc.get("probe"):
            if cks:                                          # a probe trains nothing and keeps no checkpoint
                out.append(dict(variant=v["name"], error="a probe or preflight-only run lists no checkpoint"))
            continue
        elif not cks or any(not isinstance(c, dict) or c.get("which") not in ("final", "best", "ema") for c in cks):
            out.append(dict(variant=v["name"], error="checkpoints are final, best or ema"))
            continue
        out += [dict(e, variant=v["name"]) for e in _files(dict(files=[dict(c, role="shard") for c in cks]), run_dir, ("shard",))]
    return out


def from_run(run_dir, path, expect_sha256=None, kinds=None) -> dict:
    """A hand-off file `path` (relative) of the passed run in run_dir, checked against that run's stage record."""
    rp = os.path.join(run_dir, "record.json")
    if not os.path.exists(rp):
        raise HandoffError("%s holds no run record" % run_dir)
    with open(rp) as f:
        rec = json.load(f)
    if rec.get("status") != "passed":
        raise HandoffError("run %s is %s, not passed" % (rec.get("run_id"), rec.get("status")))
    full = os.path.join(run_dir, path)
    sha = _file_sha(full)
    listed = [o for s in rec.get("stages") or [] for o in s.get("outputs") or [] if o.get("path") == path]
    if not listed or any(o.get("sha256") != sha for o in listed):
        raise HandoffError("%s of run %s is not the file its stage record lists" % (path, rec.get("run_id")))
    with open(full) as f:
        doc = json.load(f)
    if doc.get("sha256") != digest(doc):
        raise HandoffError("%s of run %s does not match its own sha256" % (path, rec.get("run_id")))
    if expect_sha256 and doc["sha256"] != expect_sha256:
        raise HandoffError("%s of run %s is manifest %s, %s expected" % (path, rec.get("run_id"), doc["sha256"], expect_sha256))
    if doc.get("format") == "rel-tableset":
        bad = sealed_problems(doc)
        if bad:
            raise HandoffError("%s of run %s: %s" % (path, rec.get("run_id"), bad[0]["error"]))
        if kinds is not None and doc.get("kind") not in kinds:
            raise HandoffError("%s of run %s is a %s table set; this reader takes %s only" % (path, rec.get("run_id"), doc.get("kind"), ", ".join(kinds)))
    return doc
