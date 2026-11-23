"""Stage rel_w3_traincheck (package W3; pipeline rel_tables, numpy only): W3's fourth validity test on the training's
own renderings, before a round trains on them (the programme lead's ruling of 2026-11-24 for round 1's third batch and
round 2). `python -m ci.stages.rel.w3_traincheck RUN_DIR`.

From the training part of a passed tables run, `tables` tables are drawn by the evaluator's 64-bit hash; for each, the
rows a training sample keeps are drawn as the round's sampler draws them (row 0 and a random share of the others, rows
4 to 32 in all), and every mask scheme of the training (data/masks: scattered in every row, a span in any row, the
whole gene) plans the sample's hidden cells. The sample is rendered as the training renders it
(seeddrift/train/rel/tables/strict.py: the correspondence aligned again with every hidden residue unknown, or the whole
gene on its frame relative; then leak-free) and the fourth test is run on it (strict.fourth_test). The cost of a
rendering is measured on the way.

Profile block rel_w3_traincheck: tables_from, tablesets [{path, sha256}], tables (1000), masks (default scattered, span,
whole_gene), rows ([4, 32]), seed (20261124), processes (6).
Outputs under data/rel_w3/traincheck/: report.json (per mask: samples, failures and the first failed tables, cells
scored, milliseconds per rendering). Gate G4_rel_w3_traincheck: every mask checked on every table drawn, no failure.
"""
from __future__ import annotations

import importlib
import json
import os
import time

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from seeddrift.train.rel import handoff
from seeddrift.train.rel.data import tables as T
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.tables import strict as ST

STAGE = "rel_w3_traincheck"
OUT = os.path.join("data", "rel_w3", "traincheck")
KEYS = {"tables_from", "tablesets", "tables", "masks", "rows", "seed", "processes"}
FIELDS = ("tok", "n", "col", "kind", "frame_len", "frame_row")
STREAM_DRAW, STREAM_ROWS = 71, 72
CHUNK = 50


def settings(cfg) -> dict:
    c = dict(cfg.get(STAGE) or {})
    unknown = set(c) - KEYS
    if unknown or not c.get("tables_from") or not c.get("tablesets"):
        raise SystemExit("%s: the profile names tables_from and tablesets (keys: %s): %s" % (STAGE, sorted(KEYS), sorted(unknown)))
    rows = [int(x) for x in (c.get("rows") or [4, 32])]
    if len(rows) != 2 or not 1 <= rows[0] <= rows[1]:
        raise SystemExit("%s: rows is [least, most]: %r" % (STAGE, rows))
    return dict(tables_from=str(c["tables_from"]), tablesets=list(c["tablesets"]), tables=int(c.get("tables", 1000)),
                masks=list(c.get("masks") or ["scattered", "span", "whole_gene"]), rows=rows, seed=int(c.get("seed", 20261124)),
                processes=max(1, int(c.get("processes", 6))))


def training_part(s):
    docs = []
    for t in s["tablesets"]:
        try:
            docs.append(handoff.from_run(s["tables_from"], t["path"], t.get("sha256")))
        except handoff.HandoffError as err:
            raise SystemExit("%s: %s" % (STAGE, err))
    sets = shards.by_split(s["tables_from"], docs)
    if shards.SPLIT_TRAIN not in sets:
        raise SystemExit("%s: the tables run has no training part" % STAGE)
    return docs, sets[shards.SPLIT_TRAIN]


def drawn(n_tables, k, seed):
    """The positions of the k tables checked, in the order of the hash of their positions."""
    order = np.argsort(cond.hash_u01(seed, STREAM_DRAW, np.array([0]), np.arange(n_tables)[None, :])[0], kind="stable")
    return np.sort(order[:min(int(k), int(n_tables))])


def rows_of(tab, t, rows, seed):
    """The rows a training sample keeps: row 0 and, of the other rows with a block, a share drawn by the hash, from
    rows[0] to rows[1] rows in all (fewer when the table has fewer)."""
    present = T.present_rows(tab)
    others = present[1:]
    u = cond.hash_u01(seed, STREAM_ROWS, np.array([int(t)]), np.arange(len(others) + 1)[None, :])[0]
    total = min(len(present), rows[0] + int(u[0] * (rows[1] - rows[0] + 1)))
    keep = others[np.argsort(u[1:], kind="stable")][:max(total - 1, 0)]
    return np.concatenate([[0], np.sort(keep)]).astype(np.int64)


def check_tables(ts, ids, masks, rows, seed):
    """Per mask: samples, failures (table positions), cells and seconds of renderings, for the tables ids of ts."""
    mods = {m: importlib.import_module("seeddrift.train.rel.data.masks." + m) for m in masks}
    out = {m: dict(samples=0, failures=[], cells=0, seconds=0.0) for m in masks}
    for t in ids:
        got = ts.load(int(t), int(t) + 1, FIELDS)
        tab = {f: got[f][0] for f in FIELDS}
        tab = T.take_rows(tab, rows_of(tab, t, rows, seed))
        for k, m in enumerate(masks):
            vis, tgt = mods[m].plan(dict(n=tab["n"], kind=tab["kind"]), np.random.default_rng([seed, int(t), k]))
            t0 = time.time()
            res = ST.fourth_test(tab, vis, tgt, whole_gene=m == "whole_gene", key=int(t) * 8 + k, seed=seed)
            r = out[m]
            r["seconds"] += time.time() - t0
            r["samples"] += 1
            r["cells"] += res["cells"]
            if not res["passed"]:
                r["failures"].append(int(t))
    return out


_SHARED = {}


def _unit(ids):
    s = _SHARED
    return check_tables(s["ts"], ids, s["masks"], s["rows"], s["seed"])


def build(st, cfg, run_dir):
    s, t0 = settings(cfg), time.time()
    docs, ts = training_part(s)
    ids = drawn(ts.tables, s["tables"], s["seed"])
    chunks = [ids[a:a + CHUNK] for a in range(0, len(ids), CHUNK)]
    if s["processes"] > 1 and len(chunks) > 1:
        import multiprocessing as mp
        _SHARED.update(ts=ts, masks=s["masks"], rows=s["rows"], seed=s["seed"])
        try:
            with mp.get_context("fork").Pool(s["processes"]) as pool:
                parts = pool.map(_unit, chunks)
        finally:
            _SHARED.clear()
    else:
        parts = [check_tables(ts, c, s["masks"], s["rows"], s["seed"]) for c in chunks]
    report = dict(version=1, strict_version=ST.VERSION, aligner_version=__import__("seeddrift.train.rel.tables.align", fromlist=["x"]).VERSION,
                  tables=int(len(ids)), rows=s["rows"], seed=s["seed"], source=dict(run=os.path.basename(os.path.normpath(s["tables_from"])),
                  tablesets=[dict(name=d["name"], sha256=d["sha256"]) for d in docs]), masks={})
    for m in s["masks"]:
        r = dict(samples=0, failures=[], cells=0, seconds=0.0)
        for p in parts:
            for k in ("samples", "cells", "seconds"):
                r[k] += p[m][k]
            r["failures"] += p[m]["failures"]
        report["masks"][m] = dict(samples=r["samples"], failures=len(r["failures"]), failed_tables=sorted(r["failures"])[:20], cells=r["cells"],
                                  ms_per_rendering=round(1000.0 * r["seconds"] / max(2 * r["samples"], 1), 2))
    report["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(run_dir, OUT, "report.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    write_json(p, report)
    st.output("report", p)
    st.extra.update(failures={m: r["failures"] for m, r in report["masks"].items()}, ms={m: r["ms_per_rendering"] for m, r in report["masks"].items()})


def gate(st, cfg, run_dir):
    s = settings(cfg)
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        report = json.load(f)
    problems = []
    for m in s["masks"]:
        r = (report.get("masks") or {}).get(m)
        if r is None or r["samples"] != report.get("tables") or not r["samples"]:
            problems.append(dict(mask=m, error="not checked on every table drawn"))
        elif r["failures"]:
            problems.append(dict(mask=m, error="the fourth test failed", failures=r["failures"], tables=r["failed_tables"]))
    require(GateResult("G4_rel_w3_traincheck", "fail" if problems else "pass", dict(problems=problems, report=report.get("masks"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        report = json.load(f)
    md = ["W3's fourth test on the training's renderings: %d tables of %s, rows %s." % (report["tables"], report["source"]["run"], report["rows"]), "",
          "| mask | samples | failures | cells scored | ms per rendering |", "|---|---|---|---|---|"]
    md += ["| %s | %d | %d | %d | %.1f |" % (m, r["samples"], r["failures"], r["cells"], r["ms_per_rendering"]) for m, r in report["masks"].items()]
    return dict(json={m: dict(failures=r["failures"], ms=r["ms_per_rendering"]) for m, r in report["masks"].items()}, md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, STAGE) as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
