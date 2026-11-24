"""Stage rel_w5_baselines: the classical baselines on the held-out tables of a passed tables run.
`python -m ci.stages.rel.w5_baselines RUN_DIR`. numpy only, no GPU.

What a model of the line has to get under: bits per codon of the hidden residues of the examined row (row 0, target
block) under the evaluation conditions of seeddrift/train/rel/evalh/conditions.py, for the baselines of
seeddrift/train/rel/baselines (own codon frequencies, the average of the rows, copy the nearest row, the copying chain
with switching). Every fitted number comes from training tables; the held-out tables are only scored.

Profile block rel_w5_baselines:
  tables_from      the run directory of the passed tables run
  tablesets        [{path, sha256}] its table-set manifests (hand-off files; together one training and one held-out set)
  frozen_fit       optional pinned output of an earlier passed baseline fit, used for scoring a held-out-only set.
                   Fields: run, commit, params_sha256, results_sha256, train_tableset_sha256, fit_tables, fit_seed.
  seed             the seed of the evaluation conditions (default: conditions.SEED)
  fit_tables       training tables the baselines are fitted on (the first of the set; default 6000)
  chain_tables     training tables the chain is fitted on (default 1500); chain: false leaves the chain out
  heldout_tables   score only the first tables of the held-out set (development; default: all)
  allow_kinds      table-set kinds accepted (default [real])
  conditions       the conditions scored (default: today's five, conditions.CONDITIONS); more of
                   conditions.EXTRA_CONDITIONS may be added, none of the five left out
  subset_per_group the subset of the evaluation during training (conditions.subset: the first tables of every group
                   in storage order; default 50, so 600 tables): its rows go to rows_subset.tsv (item_half "subset"),
                   the same cells as those tables scored alone
  offline          {run, path, sha256}: a passed run of stage w3_offline on the held-out set (W3's strict offline
                   rendering): every condition with relatives whose mask has a part there (scattered, suffix, the whole
                   gene and its re-paired condition) is scored on that part, layout leakfree, the cells and their truth
                   those of the stored tables (the evaluator's digest, checked); the conditions without relatives are
                   scored on the stored tables. The fit stays on the stored training tables.

Coverage: the groups the table set covers (its manifest says which); development diagnosis only. The limits written
into the results (LIMITS) say which way each one pulls the bits (lower is better): under the stored layout the bits
with the whole gene hidden are too low (optimistic), scattered cells only slightly (not measured), the conditions
without relatives not at all; the classical baselines are a lower bound on what the relatives are worth, not an
estimate; the choice of the rows biases that worth in a direction not known.

Outputs under data/rel_w5/baselines/: rows.tsv (the result table, columns of the contract, item_half "all"),
rows_subset.tsv (the same on the subset of the evaluation during training, item_half "subset"), params.json (the fitted
numbers), cells.npz (the bits of every scored cell, so that the strata can be formed again without scoring again: per
condition the held-out table of every cell and, per baseline, its bits and the bits of its amino acid, float32; the
group of every held-out table; results.json's "cells" names the arrays), report.md, results.json (written last: the
headline numbers, the digests of what was scored on all tables and on the subset, the inputs).
Gate G4_rel_w5_baselines: the results are complete, finite and normalised, the fit read training tables only and the
scoring held-out tables only; cells.npz is the file results.json names and gives again, from its cells, every reported group, stratum and component on the full set and its subset.
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
from ci.stages.rel import common, w5_frozen_fit
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.baselines import run as brun
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score, shards

STAGE = "rel_w5_baselines"
SUBSET = "subset"                                                # item_half of the rows on the subset of the evaluation during training
OUT = os.path.join("data", "rel_w5", "baselines")
OFFLINE_PART = {cond.SCATTERED: "scattered", "suffix": "suffix", cond.WHOLE_GENE: "whole_gene"}
LIMIT_OFFLINE = ("the conditions with relatives are read on W3's strict offline rendering (layout leakfree): no leak of the hidden "
                 "residues there, unlike the stored layout; the baselines are fitted on the stored training tables, so with the whole "
                 "gene hidden they read a layout (the frame relative) they were not fitted on")
LIMITS = [
    "lower bits are better. The stored layout shows the relatives where the hidden residues align: with the whole gene hidden the bits are "
    "too low (optimistic), the worth of the relatives there too high and the cost of re-pairing too small; with scattered cells hidden the "
    "bits are only slightly too low (not measured); the two conditions without relatives have no leak",
    "the classical baselines are weak (no codon substitution model, no amino-acid model): what they gain from the relatives is a lower bound "
    "on what the relatives are worth, not an estimate",
    "bias of unknown direction: rows drawn among genomes with a homolog row on the target gene (a similarity ranking decides that) make the "
    "relatives worth more; held-out genomes have no close relative of their own cluster and the rows are drawn at random, which makes them "
    "worth less; close relatives are not covered",
    "nearest-identity strata require at least half the target block to be covered; tables with only shorter matches have their own stratum",
    "the five eukaryote groups hold only the first 1,500 genes of a genome: they are read apart from prokaryotes, plasmids and viruses, and "
    "the headline is the tables with a relative on the target block, not the mean over all tables",
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
    if "frozen_fit" in c:
        if set(sets) != {shards.SPLIT_HELDOUT} or any(d.get("purpose") != "heldout" or d.get("not_for_training") is not True for d in docs):
            raise SystemExit(f"{STAGE}: frozen_fit requires heldout-only tables marked not_for_training")
    elif sorted(sets) != [shards.SPLIT_TRAIN, shards.SPLIT_HELDOUT]:
        raise SystemExit("%s: one training and one held-out table set are needed, found splits %s" % (STAGE, sorted(sets)))
    return src, docs, sets


def offline_parts(c, held, n_held):
    """None, or the strict offline rendering of the held-out set the profile names: its run, manifest and parts."""
    o = c.get("offline")
    if not o:
        return None
    try:
        doc = handoff.from_run(str(o["run"]), str(o["path"]), o.get("sha256"))
    except handoff.HandoffError as err:
        raise SystemExit("%s: %s" % (STAGE, err))
    names = {t.get("sha256") for t in (doc.get("source") or {}).get("tablesets") or []}
    if held.manifest["sha256"] not in names:
        raise SystemExit("%s: the offline rendering %s is not of the held-out set read here" % (STAGE, doc.get("name")))
    parts = shards.parts(str(o["run"]), doc)
    for name, ts in parts.items():
        if ts.tables < n_held:
            raise SystemExit("%s: offline part %s holds %d tables, %d are scored" % (STAGE, name, ts.tables, n_held))
    return dict(run=str(o["run"]), doc=doc, parts=parts)


def source_of(condition, offline):
    """The offline part a condition is scored on, or None for the stored tables."""
    mask, mode = cond.parse(condition)
    if not offline or mode == cond.NO_RELATIVES:
        return None
    part = OFFLINE_PART.get(mask)
    return part if part in offline["parts"] else None


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


def write_cells(path, results, group):
    """cells.npz: per condition (in results' order) c<i>_table, the held-out table (position) of every scored cell, and
    per baseline c<i>_b<j> its bits and c<i>_a<j> the bits of the amino acid (not for the joint reading), float32;
    group, the group index of every held-out table. Returns the index results.json keeps: {condition: dict(table,
    cells, baselines {name: dict(bits, amino_acid_bits)})}."""
    arrays, index = dict(group=np.asarray(group).astype(np.int16)), {}
    for i, (c, r) in enumerate(results.items()):
        entry = dict(table="c%d_table" % i, cells=int(r["cell_table"].shape[0]), baselines={})
        arrays[entry["table"]] = np.asarray(r["cell_table"]).astype(np.int32)
        for j, (n, b) in enumerate(r["bits"].items()):
            e = dict(bits="c%d_b%d" % (i, j))
            arrays[e["bits"]] = np.asarray(b, dtype=np.float32)
            if n in (r.get("amino_acid_bits") or {}):
                e["amino_acid_bits"] = "c%d_a%d" % (i, j)
                arrays[e["amino_acid_bits"]] = np.asarray(r["amino_acid_bits"][n], dtype=np.float32)
            entry["baselines"][n] = e
        index[c] = entry
    np.savez(path, **arrays)
    return index


def audit_rows(path, doc, cfg, rows):
    """Recompute the full and subset TSVs from the saved cells and the pinned held-out shards."""
    c = block(cfg)
    src, _, sets = inputs(cfg)
    held = sets[shards.SPLIT_HELDOUT]
    count = min(int(c.get("heldout_tables") or held.tables), held.tables)
    meta = held.load(0, count, ("group", "identity", "n", "col", "kind"))
    group = meta["group"]
    strata = score.strata_of(meta["identity"], brun.other_rows(meta), meta["kind"], score.row_coverage(meta["n"], meta["col"]))
    del meta
    names = dict(enumerate(shards.group_names(src, held.manifest)))
    n_groups = max(len(names), int(group.max()) + 1)
    subset = cond.subset(group, int(c.get("subset_per_group", cond.SUBSET_PER_GROUP)))
    expected = {}
    for row in rows:
        key = (row["condition"], row["variant"], row["item_half"], row["group"], row["stratum"])
        if key in expected:
            raise ValueError(f"duplicate row {key}")
        expected[key] = row
    index = doc["cells"]["conditions"]
    conditions = c.get("conditions") or cond.CONDITIONS
    if set(index) != set(conditions) or doc["cells"]["score_version"] != score.VERSION:
        raise ValueError("the saved cells disagree with the configured conditions or scoring version")

    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(archive["group"], group):
            raise ValueError("group index disagrees with the held-out shards")
        for condition in conditions:
            entry = index[condition]
            table = archive[entry["table"]]
            if (table.shape != (entry["cells"],) or table.dtype.kind not in "iu"
                    or (table < 0).any() or (table >= count).any()):
                raise ValueError(f"invalid table indices under {condition}")
            mask = np.isin(table, subset)
            baselines = brun.names_under(condition, c.get("chain", True))
            if set(entry["baselines"]) != set(baselines):
                raise ValueError(f"baseline index disagrees with the configured baselines under {condition}")
            for name in baselines:
                keys = entry["baselines"][name]
                fields = {"bits"} if name == brun.switch.JOINT else {"bits", "amino_acid_bits"}
                if set(keys) != fields:
                    raise ValueError(f"incorrect components for {condition}, {name}")
                bits = archive[keys["bits"]]
                aa = archive[keys["amino_acid_bits"]] if "amino_acid_bits" in fields else None
                for values in (bits,) if aa is None else (bits, aa):
                    if values.shape != table.shape or not np.isfinite(values).all() or (values < -1e-5).any():
                        raise ValueError(f"invalid per-cell scores for {condition}, {name}")
                if aa is not None and (aa > bits + 1e-5).any():
                    raise ValueError(f"amino-acid score exceeds the codon score for {condition}, {name}")
                for half, take in ((score.ALL, slice(None)), (SUBSET, mask)):
                    part = None if aa is None else aa[take].astype(np.float64)
                    recomputed = brun.cell_rows(bits[take].astype(np.float64), part, table[take], group, strata, n_groups, names)
                    for result in recomputed:
                        key = (condition, name, half, result["group"], result["stratum"])
                        stored = expected.pop(key, None)
                        if (stored is None or not math.isclose(float(stored["bits"]), result["bits"], abs_tol=1e-4, rel_tol=0)
                                or int(stored["target_cells"]) != result["target_cells"] or int(stored["n_items"]) != result["n_items"]):
                            raise ValueError(f"the cells give {result['bits']:.6f} bits on {result['target_cells']} cells "
                                             f"and {result['n_items']} tables for {key}, inconsistent with the TSV")
    if expected:
        raise ValueError(f"unexpected result row {next(iter(expected))}")



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
        held = sets[shards.SPLIT_HELDOUT]
        n_held = min(int(c.get("heldout_tables") or held.tables), held.tables)
        with_chain = c.get("chain", True) is not False
        conditions = list(c.get("conditions") or cond.CONDITIONS)
        try:
            [cond.parse(x) for x in conditions]
        except ValueError as err:
            raise SystemExit("%s: %s" % (STAGE, err))
        if [x for x in cond.CONDITIONS if x not in conditions] or len(set(conditions)) != len(conditions):
            raise SystemExit("%s: the conditions hold today's five (%s), each once" % (STAGE, ", ".join(cond.CONDITIONS)))
        frozen_bytes = None
        if "frozen_fit" in c:
            params, frozen_bytes, source_fit = w5_frozen_fit.load_frozen_fit(c["frozen_fit"])
            if seed + 1 != source_fit["fit_seed"]:
                raise SystemExit(f"{STAGE}: use the scoring seed paired with the original fit")
            if any(mask not in params or (with_chain and "chains" not in params[mask]) for mask in brun.masks_of(conditions)):
                raise SystemExit(f"{STAGE}: the frozen fit does not cover the configured masks and baselines")
            fit_record = dict(tables=source_fit["fit_tables"], seed=source_fit["fit_seed"], source=source_fit)
            log(f"scoring {n_held} held-out tables with the frozen parameters of {source_fit['run_id']}")
        else:
            train = sets[shards.SPLIT_TRAIN]
            n_fit = min(int(c.get("fit_tables", 6000)), train.tables)
            log(f"fitting on {n_fit} training tables, scoring {n_held} held-out tables")
            fit_split = train.load(0, n_fit, ("split",))["split"]
            if (fit_split != shards.SPLIT_TRAIN).any():
                raise SystemExit(f"{STAGE}: a table to fit on is not a training table")
            params = brun.fit_all(train.load(0, n_fit, shards.MODEL_FIELDS), np.arange(n_fit), seed + 1,
                                  chain_tables=int(c.get("chain_tables", 1500)), with_chain=with_chain,
                                  log=log, masks=brun.masks_of(conditions))
            fit_record = dict(tables=n_fit, seed=seed + 1, set=train.manifest["name"], folder=train.folder, files_read=len(train.verified))

        tables = held.load(0, n_held, shards.MODEL_FIELDS)
        meta = held.load(0, n_held, ("group", "split", "identity"))
        offline = offline_parts(c, held, n_held)
        source = {x: source_of(x, offline) for x in conditions}
        layouts = {x: (api.LAYOUT_LEAKFREE if source[x] else api.LAYOUT_STORED) for x in conditions}
        if frozen_bytes is not None:
            for name in conditions:
                if layouts[name] != source_fit["original_layouts"].get(name):
                    raise SystemExit(f"{STAGE}: layout differs from the pinned original under {name}")
        if (meta["split"] != shards.SPLIT_HELDOUT).any():
            raise SystemExit("%s: a table to score is not a held-out table" % STAGE)
        groups = shards.group_names(src, held.manifest)
        n_groups = max(len(groups), int(meta["group"].max()) + 1)
        sub = cond.subset(meta["group"], int(c.get("subset_per_group", cond.SUBSET_PER_GROUP)))
        results = {}
        for part in sorted({v for v in source.values() if v}) + [None]:     # each source's conditions on its tables
            todo = [x for x in conditions if source[x] == part]
            if not todo:
                continue
            tabs = tables if part is None else offline["parts"][part].load(0, n_held, shards.MODEL_FIELDS)
            results.update(brun.evaluate(tabs, np.arange(n_held), params, conditions=todo, seed=seed, with_chain=with_chain, log=log, subset=sub))
            if part is not None:
                del tabs
        results = {x: results[x] for x in conditions}
        if frozen_bytes is not None:
            # ItemsDigest includes the companion-row permutation. Adding rows changes it for the
            # repaired control; that control needs its own paired eligibility check at report time.
            for name, result in results.items():
                if cond.parse(name)[1] != cond.REPAIRED and result["items_sha256"] != source_fit["original_items"][name]["sha256"]:
                    raise SystemExit(f"{STAGE}: the paired target digest differs from the frozen source under {name}")
        if offline:                                                     # the cells scored on a part are those of the stored tables
            for x in conditions:
                if source[x]:
                    check = cond.ItemsDigest(x, seed)
                    for a in range(0, n_held, 512):
                        check.add(cond.materialise({f: tables[f][a:a + 512] for f in ("tok", "n", "col", "kind")}, np.arange(a, min(a + 512, n_held)), x, seed))
                    if check.hexdigest() != results[x]["items_sha256"]:
                        raise SystemExit("%s: under %s the offline part scored other cells than the stored tables" % (STAGE, x))
        head = dict(run=os.path.basename(os.path.normpath(run_dir)), commit=(cfg.get("git") or {}).get("sha", ""), seed=seed,
                    table_set=held.manifest["name"], table_kind=held.manifest["kind"], items_sha256=held.manifest["sha256"],
                    versions="evalh.conditions=%d;evalh.score=%d;baselines.run=%d;table=%s" % (cond.VERSION, score.VERSION, brun.VERSION, held.manifest["table_version"]))
        coverage = score.row_coverage(tables["n"], tables["col"])           # score version 5: the rows that count for the nearest identity
        rows = brun.result_rows(results, meta["group"], meta["identity"], params, head, n_groups, dict(enumerate(groups)), brun.other_rows(tables),
                                kind=tables["kind"], layouts=layouts, coverage=coverage)
        on_subset = brun.restrict(results, sub)
        rows_sub = brun.result_rows(on_subset, meta["group"], meta["identity"], params, head, n_groups, dict(enumerate(groups)), brun.other_rows(tables),
                                    item_half=SUBSET, kind=tables["kind"], layouts=layouts, coverage=coverage)
        fine = score.fine_stratum(meta["identity"], tables["kind"], coverage)
        cov_by = score.coverage_by_stratum(fine, score.nearest_coverage(meta["identity"], coverage, tables["kind"]))

        rows_path, sub_path, params_path, report_path, out_path, cells_path = (st.path("rel_w5", "baselines", n) for n in
                                                                               ("rows.tsv", "rows_subset.tsv", "params.json", "report.md", "results.json", "cells.npz"))
        score.write_rows(rows_path, rows)
        cells_index = write_cells(cells_path, results, meta["group"])
        score.write_rows(sub_path, rows_sub)
        if frozen_bytes is None:
            write_json(params_path, params)
        else:
            with open(params_path, "wb") as out:
                out.write(frozen_bytes)
        title = "Classical baselines on table set %s (%d held-out tables, seed %d)" % (held.manifest["name"], n_held, seed)
        with open(report_path, "w") as f:
            f.write(brun.markdown(rows, title, groups, cov_by) + brun.subset_markdown(rows_sub, sub.shape[0]) + "\nLimits:\n"
                    + "".join("- %s\n" % x for x in ([LIMIT_OFFLINE] if offline else []) + LIMITS))
        doc = dict(
            stage=STAGE, api=api.API, seed=seed, conditions=list(results), baselines=list(brun.names(with_chain)), chain=with_chain,
            reference=brun.REFERENCE,
            headline=headline(rows), groups=groups, n_groups=n_groups, limits=([LIMIT_OFFLINE] if offline else []) + LIMITS, layouts=layouts,
            offline=(dict(run_id=os.path.basename(os.path.normpath(offline["run"])), name=offline["doc"]["name"], sha256=offline["doc"]["sha256"],
                          parts={x: source[x] for x in conditions if source[x]}) if offline else None),
            items={k: dict(sha256=r["items_sha256"], cells=int(r["cell_table"].shape[0]), tables=r["tables"]) for k, r in results.items()},
            subset=dict(per_group=int(c.get("subset_per_group", cond.SUBSET_PER_GROUP)), tables=int(sub.shape[0]), rows=len(rows_sub),
                        rows_sha256=sha256_file(sub_path), bits=brun.summary(rows_sub)[0],
                        items={k: dict(sha256=r["items_sha256"], cells=int(r["cell_table"].shape[0]), tables=r["tables"]) for k, r in on_subset.items()}),
            selfcheck=cond.selfcheck(), numpy=np.__version__,
            fit=dict(**fit_record,
                     numbers={m: {k: v for k, v in p.items() if k != "prior"} for m, p in params.items()}),
            heldout=dict(tables=n_held, set=held.manifest["name"], folder=held.folder, files_read=len(held.verified), kind=held.manifest["kind"],
                         coverage=held.manifest.get("coverage")),
            tables_run=dict(run_id=os.path.basename(os.path.normpath(src)), tablesets=[dict(name=d["name"], sha256=d["sha256"], tables=d["tables"]) for d in docs]),
            sealed=sealed, rows=len(rows), rows_sha256=sha256_file(rows_path), seconds=round(time.time() - t0, 1),
            cells=dict(file="cells.npz", sha256=sha256_file(cells_path, use_sidecar=False), dtype="float32", score_version=score.VERSION,
                       conditions=cells_index),
        )
        write_json(out_path, doc)
        for name, p in (("rows", rows_path), ("rows_subset", sub_path), ("params", params_path), ("cells", cells_path), ("report", report_path),
                        ("results", out_path)):
            st.output(name, p)
        st.extra.update(tables_scored=n_held, tables_fitted=0 if frozen_bytes is not None else fit_record["tables"], rows=len(rows))
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
    if len(rows) != doc.get("rows") or any(r["item_half"] != score.ALL for r in rows):
        problems.append(dict(error="rows.tsv has an incorrect row count or contains subset rows"))
    seen = {}
    for r in rows:
        try:
            b = float(r["bits"])
        except ValueError:
            b = float("nan")
        want = (doc.get("layouts") or {}).get(r["condition"], api.LAYOUT_STORED)       # leakfree where read on the strict offline part
        if not math.isfinite(b) or b < 0 or int(r["target_cells"]) <= 0 or r["status"] != "ok" or r["layout"] != want:
            problems.append(dict(row=[r["variant"], r["condition"], r["group"], r["stratum"]], error="bits not finite, no cell, or not marked ok and in its layout"))
        if r["group"] == score.ALL and r["stratum"] == score.ALL:
            seen[(r["condition"], r["variant"])] = b
    for c in cond.CONDITIONS:
        if c not in doc.get("conditions", []):
            problems.append(dict(condition=c, error="condition not evaluated"))
    for c in list(dict.fromkeys(list(cond.CONDITIONS) + list(doc.get("conditions") or []))):
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
    sub = doc.get("subset") or {}
    sub_rows = []
    sub_path = os.path.join(folder, "rows_subset.tsv")
    if not sub or not os.path.exists(sub_path) or sha256_file(sub_path, use_sidecar=False) != sub.get("rows_sha256"):
        problems.append(dict(error="rows_subset.tsv is missing or not the file results.json names"))
    else:
        sub_rows = score.read_rows(sub_path)
        if len(sub_rows) != sub.get("rows"):
            problems.append(dict(error="rows_subset.tsv has an incorrect row count"))
        top = {(r["condition"], r["variant"]) for r in sub_rows if r["group"] == score.ALL and r["stratum"] == score.ALL}
        if any(r["item_half"] != SUBSET for r in sub_rows) or not all(math.isfinite(float(r["bits"])) for r in sub_rows):
            problems.append(dict(error="a subset row is not marked subset or its bits are not finite"))
        for c in doc.get("conditions") or []:
            missing = [n for n in brun.names_under(c, bool(doc.get("chain"))) if (c, n) not in top]
            if missing or not (sub.get("items") or {}).get(c, {}).get("sha256"):
                problems.append(dict(condition=c, error="the subset lacks rows over all groups or the digest of its cells", baselines=missing))
    cells = doc.get("cells") or {}
    cells_path = os.path.join(folder, "cells.npz")
    if not cells or not os.path.exists(cells_path) or sha256_file(cells_path, use_sidecar=False) != cells.get("sha256"):
        problems.append(dict(error="cells.npz is missing or not the file results.json names"))
    else:
        if sorted(cells.get("conditions") or {}) != sorted(doc.get("conditions") or []):
            problems.append(dict(error="cells.npz does not hold every condition scored"))
        try:
            audit_rows(cells_path, doc, cfg, rows + sub_rows)
        except (KeyError, IndexError, TypeError, ValueError, OSError) as err:
            problems.append(dict(error=str(err)))
    fit_spec = block(cfg).get("frozen_fit")
    if fit_spec is not None:
        params_path = os.path.join(folder, "params.json")
        if (not os.path.isfile(params_path) or sha256_file(params_path, use_sidecar=False) != fit_spec["params_sha256"]
                or (doc.get("fit", {}).get("source") or {}).get("params_sha256") != fit_spec["params_sha256"]):
            problems.append(dict(error="scored parameters do not match the pinned fit"))
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
              bits_subset={c: {n: r4(b) for n, b in v.items() if n in SUMMARY_BASELINES} for c, v in ((doc.get("subset") or {}).get("bits") or {}).items()},
              cells={k: v["cells"] for k, v in doc["items"].items()}, seed=doc["seed"], heldout_tables=doc["heldout"]["tables"],
              fit_tables=doc["fit"]["tables"], table_set=doc["heldout"]["set"], tables_run=doc["tables_run"]["run_id"], limits=doc["limits"])
    return dict(json=js, md=md)


if __name__ == "__main__":
    main()
