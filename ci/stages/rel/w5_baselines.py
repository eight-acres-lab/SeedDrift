"""Stage rel_w5_baselines: the classical baselines on the held-out tables of a passed tables run.
`python -m ci.stages.rel.w5_baselines RUN_DIR`. numpy only, no GPU.

What a model of the line has to get under: bits per codon of the hidden residues of the examined row (row 0, target
block) under the evaluation conditions of seeddrift/train/rel/evalh/conditions.py, for the baselines of
seeddrift/train/rel/baselines (own codon frequencies, the average of the rows, copy the nearest row, the copying chain
with switching). Every fitted number comes from training tables; the held-out tables are only scored.

Profile block rel_w5_baselines:
  tables_from      the run directory of the passed tables run
  tablesets        [{path, sha256}] its table-set manifests (hand-off files; together one training and one held-out set)
  seed             the seed of the evaluation conditions (default: conditions.SEED)
  fit_tables       training tables the baselines are fitted on (the first of the set; default 6000)
  chain_tables     training tables the chain is fitted on (default 1500); chain: false leaves the chain out
  heldout_tables   score only the first tables of the held-out set (development; default: all)
  allow_kinds      table-set kinds accepted (default [real])

Coverage: the groups the table set covers (its manifest says which); development diagnosis only. Limits written into
the results: the stored layout places the relatives by the hidden residues, so every number with the whole gene hidden
is an upper bound; the rows of a table are drawn among the genomes that have a homolog row on the target gene, so what
the relatives are worth is overestimated; one seed, no noise floor.

Outputs under data/rel_w5/baselines/: rows.tsv (the result table, columns of the contract), params.json (the fitted
numbers), report.md, results.json (written last: the headline numbers, the digests of what was scored, the inputs).
Gate G4_rel_w5_baselines: the results are complete, finite and normalised, the fit read training tables only and the
scoring held-out tables only.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.baselines import run as brun
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score, shards

STAGE = "rel_w5_baselines"
OUT = os.path.join("data", "rel_w5", "baselines")
LIMITS = [
    "stored layout: the relatives are placed by the hidden residues, so the numbers with the whole gene hidden are an upper bound",
    "rows are drawn among genomes with a homolog row on the target gene (a similarity ranking decides that), so the worth of the relatives is overestimated",
    "one seed, no noise floor: read large effects only",
    "development diagnosis, not a selection round",
]


def block(cfg):
    c = dict(cfg.get(STAGE) or {})
    if not c.get("tables_from") or not c.get("tablesets"):
        raise SystemExit("%s: the profile names tables_from and tablesets" % STAGE)
    return c


def inputs(cfg):
    """The manifests of the tables run, checked against that run's record, and its two table sets by split."""
    c = block(cfg)
    src = c["tables_from"]
    allow = list(c.get("allow_kinds") or [api.TABLE_REAL])
    docs = []
    for t in c["tablesets"]:
        try:
            doc = handoff.from_run(src, t["path"], t.get("sha256"))
        except handoff.HandoffError as err:
            raise SystemExit("%s: %s" % (STAGE, err))
        bad = handoff.tableset_problems(doc)
        if bad:
            raise SystemExit("%s: %s is not a valid table-set manifest: %s" % (STAGE, t["path"], bad[:3]))
        if doc["kind"] not in allow:
            raise SystemExit("%s: table set %s is %s; this profile accepts %s" % (STAGE, doc["name"], doc["kind"], allow))
        docs.append(doc)
    try:
        sets = shards.by_split(src, docs)
    except ValueError as err:
        raise SystemExit("%s: %s" % (STAGE, err))
    if sorted(sets) != [shards.SPLIT_TRAIN, shards.SPLIT_HELDOUT]:
        raise SystemExit("%s: one training and one held-out table set are needed, found splits %s" % (STAGE, sorted(sets)))
    return src, docs, sets


def check_genomes(cfg, docs):
    """The sealed check on every genome of the table sets that are read (real and ruler sets list them)."""
    genomes = {}
    for d in docs:
        for g in d.get("genomes") or []:
            genomes[g["id"]] = g
    if not genomes:
        return dict(checked=0, note="no genomes listed (synthetic tables)")
    return common.sealed_check(cfg, list(genomes.values()), splits={k: g.get("split") for k, g in genomes.items()})


def headline(rows):
    main, strata = brun.summary(rows)
    return dict(bits=main, by_nearest_identity=strata)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, STAGE) as st:
        cfg = st.config
        c = block(cfg)
        t0 = time.time()
        log = lambda msg: print("[%s %6.0fs] %s" % (STAGE, time.time() - t0, msg), flush=True)      # noqa: E731
        src, docs, sets = inputs(cfg)
        sealed = check_genomes(cfg, docs)
        seed = int(c.get("seed", cond.SEED))
        train, held = sets[shards.SPLIT_TRAIN], sets[shards.SPLIT_HELDOUT]
        n_fit = min(int(c.get("fit_tables", 6000)), train.tables)
        n_held = min(int(c.get("heldout_tables") or held.tables), held.tables)
        with_chain = c.get("chain", True) is not False
        log("training set %s: %d tables (fit on %d); held-out set %s: %d tables (scoring %d)" % (train.folder, train.tables, n_fit, held.folder, held.tables, n_held))

        fit_tables = train.load(0, n_fit, shards.MODEL_FIELDS)
        fit_split = train.load(0, n_fit, ("split",))["split"]
        if (fit_split != shards.SPLIT_TRAIN).any():
            raise SystemExit("%s: a table to fit on is not a training table" % STAGE)
        # the fit uses another seed than the scoring, so no cell pattern of the held-out plan is reused
        params = brun.fit_all(fit_tables, np.arange(n_fit), seed + 1, chain_tables=int(c.get("chain_tables", 1500)), with_chain=with_chain, log=log)
        del fit_tables

        tables = held.load(0, n_held, shards.MODEL_FIELDS)
        meta = held.load(0, n_held, ("group", "split", "identity"))
        if (meta["split"] != shards.SPLIT_HELDOUT).any():
            raise SystemExit("%s: a table to score is not a held-out table" % STAGE)
        groups = shards.group_names(src, held.manifest)
        n_groups = max(len(groups), int(meta["group"].max()) + 1)
        results = brun.evaluate(tables, np.arange(n_held), params, seed=seed, with_chain=with_chain, log=log)
        head = dict(run=os.path.basename(os.path.normpath(run_dir)), commit=(cfg.get("git") or {}).get("sha", ""), seed=seed,
                    table_set=held.manifest["name"], table_kind=held.manifest["kind"], items_sha256=held.manifest["sha256"],
                    versions="evalh.conditions=%d;evalh.score=%d;baselines.run=%d;table=%s" % (cond.VERSION, score.VERSION, brun.VERSION, held.manifest["table_version"]))
        rows = brun.result_rows(results, meta["group"], meta["identity"], params, head, n_groups, dict(enumerate(groups)), brun.other_rows(tables))

        rows_path, params_path, report_path, out_path = (st.path("rel_w5", "baselines", n) for n in ("rows.tsv", "params.json", "report.md", "results.json"))
        score.write_rows(rows_path, rows)
        write_json(params_path, params)
        title = "Classical baselines on table set %s (%d held-out tables, seed %d)" % (held.manifest["name"], n_held, seed)
        with open(report_path, "w") as f:
            f.write(brun.markdown(rows, title, groups) + "\nLimits:\n" + "".join("- %s\n" % x for x in LIMITS))
        doc = dict(
            stage=STAGE, api=api.API, seed=seed, conditions=list(results), baselines=list(brun.names(with_chain)), chain=with_chain,
            reference=brun.REFERENCE,
            headline=headline(rows), groups=groups, n_groups=n_groups, limits=LIMITS,
            items={k: dict(sha256=r["items_sha256"], cells=int(r["cell_table"].shape[0]), tables=r["tables"]) for k, r in results.items()},
            selfcheck=cond.selfcheck(), numpy=np.__version__,
            fit=dict(tables=n_fit, seed=seed + 1, set=train.manifest["name"], folder=train.folder, files_read=len(train.verified),
                     numbers={m: {k: v for k, v in p.items() if k != "prior"} for m, p in params.items()}),
            heldout=dict(tables=n_held, set=held.manifest["name"], folder=held.folder, files_read=len(held.verified), kind=held.manifest["kind"],
                         coverage=held.manifest.get("coverage")),
            tables_run=dict(run_id=os.path.basename(os.path.normpath(src)), tablesets=[dict(name=d["name"], sha256=d["sha256"], tables=d["tables"]) for d in docs]),
            sealed=sealed, rows=len(rows), rows_sha256=sha256_file(rows_path), seconds=round(time.time() - t0, 1),
        )
        write_json(out_path, doc)
        for name, p in (("rows", rows_path), ("params", params_path), ("report", report_path), ("results", out_path)):
            st.output(name, p)
        st.extra.update(tables_scored=n_held, tables_fitted=n_fit, rows=len(rows))
        log("done: %d rows" % len(rows))


def gate(st, cfg, run_dir):
    """G4_rel_w5_baselines on what the stage wrote."""
    folder = os.path.join(run_dir, OUT)
    with open(os.path.join(folder, "results.json")) as f:
        doc = json.load(f)
    rows = score.read_rows(os.path.join(folder, "rows.tsv"))
    problems = []
    if sha256_file(os.path.join(folder, "rows.tsv"), use_sidecar=False) != doc.get("rows_sha256"):
        problems.append(dict(error="rows.tsv is not the file results.json names"))
    seen = {}
    for r in rows:
        try:
            b = float(r["bits"])
        except ValueError:
            b = float("nan")
        if not math.isfinite(b) or b < 0 or int(r["target_cells"]) <= 0 or r["status"] != "ok" or r["layout"] != api.LAYOUT_STORED:
            problems.append(dict(row=[r["variant"], r["condition"], r["group"], r["stratum"]], error="bits not finite, no cell, or not marked ok and stored"))
        if r["group"] == score.ALL and r["stratum"] == score.ALL:
            seen[(r["condition"], r["variant"])] = b
    for c in cond.CONDITIONS:
        if c not in doc.get("conditions", []):
            problems.append(dict(condition=c, error="condition not evaluated"))
        for n in brun.names_under(c, bool(doc.get("chain"))):
            if (c, n) not in seen:
                problems.append(dict(condition=c, baseline=n, error="no row over all groups"))
        it = (doc.get("items") or {}).get(c) or {}
        if not it.get("sha256") or not it.get("cells"):
            problems.append(dict(condition=c, error="no digest of the scored cells"))
    uni = math.log2(api.TARGET_MAX)
    for (c, n), b in seen.items():
        if n == "uniform" and abs(b - uni) > 1e-6:
            problems.append(dict(condition=c, error="the uniform line is %.6f bits, not log2(%d)" % (b, api.TARGET_MAX)))
    for a, b in ((cond.SCATTERED, cond.SCATTERED + "+" + cond.NO_RELATIVES), (cond.WHOLE_GENE, cond.WHOLE_GENE + "+" + cond.NO_RELATIVES)):
        ia, ib = (doc.get("items") or {}).get(a) or {}, (doc.get("items") or {}).get(b) or {}
        if ia.get("cells") != ib.get("cells") or abs(seen.get((a, "own_freq"), 0) - seen.get((b, "own_freq"), 1)) > 1e-9:
            problems.append(dict(conditions=[a, b], error="the same cells must be scored with and without relatives"))
    if doc.get("heldout", {}).get("kind") == api.TABLE_REAL and doc.get("n_groups") != 12:
        problems.append(dict(error="a real table set has 12 groups, the results have %s" % doc.get("n_groups")))
    if not doc.get("fit", {}).get("tables") or not doc.get("heldout", {}).get("tables"):
        problems.append(dict(error="no table fitted on or scored"))
    require(GateResult("G4_rel_w5_baselines", "fail" if problems else "pass",
                       dict(problems=problems[:50], rows=len(rows), conditions=doc.get("conditions"), baselines=doc.get("baselines"),
                            tables=doc.get("heldout", {}).get("tables"), selfcheck=doc.get("selfcheck"))), run_dir)
    return sha256_file(os.path.join(folder, "results.json"), use_sidecar=False)


SUMMARY_BASELINES = ("own_freq", "row_average", "copy_nearest", "copy_nearest_binned", "switch_copy", "switch_copy.joint")


def summary(run_dir):
    """The headline numbers for the run summary, small (the summary hook takes at most 20 kB of a stage): bits of every
    baseline under every condition, and for the two main conditions the key baselines by the identity of the nearest
    row; the full tables are in report.md and rows.tsv."""
    folder = os.path.join(run_dir, OUT)
    with open(os.path.join(folder, "results.json")) as f:
        doc = json.load(f)
    with open(os.path.join(folder, "report.md")) as f:
        md = f.read().splitlines()
    head = doc["headline"]
    r4 = lambda x: round(float(x), 4)                                          # noqa: E731
    strata = {c: {s.split("=")[-1]: {n: r4(b) for n, b in v.items() if n in SUMMARY_BASELINES} for s, v in head["by_nearest_identity"].get(c, {}).items()
                  if s in score.FINE} for c in (cond.SCATTERED, cond.WHOLE_GENE)}
    js = dict(bits={c: {n: r4(b) for n, b in v.items()} for c, v in head["bits"].items()}, by_nearest_identity=strata,
              cells={k: v["cells"] for k, v in doc["items"].items()}, seed=doc["seed"], heldout_tables=doc["heldout"]["tables"],
              fit_tables=doc["fit"]["tables"], table_set=doc["heldout"]["set"], tables_run=doc["tables_run"]["run_id"], limits=doc["limits"])
    return dict(json=js, md=md)


if __name__ == "__main__":
    main()
