"""Stage rel_w2_census (package W2; pipeline rel_tables): the data census of the tables of version 0
(seeddrift/train/rel/tables/v0_census.py). `python -m ci.stages.rel.w2_census RUN_DIR`. No GPU, no genome opened: it
reads the counts stage rel_w2_v0 wrote while it built the tables (data/rel_w2/v0/counts.json, checked against the
table-set manifest), of the same run or, with rel_w2_census.tables_from, of a passed run.

Outputs under data/rel_w2/census/:
  report.json  codons per group and split; per table set and group the other rows per table, the share of their cells
               that hold codons (by role of the block; against the rows that exist and against all row slots), the
               identity of the nearest other row per block, and the share of cells that would hold codons if rows were
               drawn among the training genomes without looking at which of them have a homolog row; `headline` holds
               the four numbers of the first read-outs
               `strata` holds, per table set, the nearest other row of every table at the target locus in the
               evaluator's strata (seeddrift/train/rel/evalh/score.py: ten equal bins and none, and the trainer's six),
               read from the metadata field identity through the evaluator's reader, and the share of tables whose
               nearest row is at least 0.98 identical
               `cap` holds, per split and group of the open genomes, the genes whose homolog rows in stage pairs reach
               the cap of that stage (the `genomes` nearest genomes kept per gene), read from the `off` arrays of the
               source run's homolog rows (each file checked against the source's manifest; the sealed check first)
  report.md    the same on one page
Configuration (profile block rel_w2_census): tables_from (a passed run whose stage rel_w2_v0 wrote the tables; default:
this run), processes (for checking the source files, default 2).
Gate G4_rel_w2_census (gate()): the report covers every part of the table set with its number of tables, every share is
a share, every group of the table set has its codon counts, and the page is there.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from seeddrift.train.rel import handoff
from seeddrift.train.rel.evalh import score as SC, shards as SH
from seeddrift.train.rel.tables import v0_census as C

OUT = os.path.join("data", "rel_w2", "census")
TABLESET = os.path.join("data", "rel_w2", "v0", "tableset.json")
CAVEATS = [
    "Rows of these tables were drawn among the training genomes that have a homolog row on the target gene. Before the corpus is searched again "
    "without caps, having a homolog row is decided by a similarity ranking (stage pairs keeps the 64 nearest genomes of a gene), so the shares of "
    "filled cells and the identities overstate what a relative drawn by genome distance would give.",
    "A missing homolog row is not a missing homolog: the cell kind is 'not searched'.",
    "Identity is the share of identical codons over the codons of a row that have a column in the examined row (the stored placement).",
    "Only the first 1,500 genes of a eukaryote and the first codons of a gene up to the source's limit are in the source arrays.",
    "Development diagnosis on the train and held-out splits; the sealed test split was not opened.",
]


def tables(run_dir, cfg):
    """(run directory that holds the tables, the table-set manifest, the counts document)."""
    src = (cfg.get("rel_w2_census") or {}).get("tables_from")
    unknown = set(cfg.get("rel_w2_census") or {}) - {"tables_from", "processes"}
    if unknown:
        raise SystemExit("rel_w2_census: unknown keys %s" % sorted(unknown))
    if src:
        try:
            doc = handoff.from_run(src, TABLESET)
        except handoff.HandoffError as err:
            raise SystemExit("rel_w2_census: %s" % err)
    else:
        src = run_dir
        with open(os.path.join(run_dir, TABLESET)) as f:
            doc = json.load(f)
        if doc.get("sha256") != handoff.digest(doc):
            raise SystemExit("rel_w2_census: the table-set manifest of this run does not match its own sha256")
    cf = [f for f in doc.get("files") or [] if f.get("role") == "meta" and os.path.basename(f["path"]) == "counts.json"]
    if len(cf) != 1 or sha256_file(os.path.join(src, cf[0]["path"]), use_sidecar=False) != cf[0]["sha256"]:
        raise SystemExit("rel_w2_census: the counts of the table set are missing or not the file its manifest names")
    with open(os.path.join(src, cf[0]["path"])) as f:
        return src, doc, json.load(f)


HIGH = (0.98, 0.99, 1.0)                     # thresholds of the nearest identity reported on their own
EPS = 1e-6                                   # identities are stored as float32


def _counts(idx, labels, n):
    c = np.bincount(np.asarray(idx, dtype=np.int64), minlength=len(labels))[:len(labels)]
    return {lab: dict(tables=int(c[k]), share=float(c[k]) / n if n else None) for k, lab in enumerate(labels)}


def strata(src, doc, groups) -> dict:
    """Per table set (train, heldout): the nearest other row of every table at the target locus (the largest identity
    of rows 1..) in the evaluator's strata, overall and per group, and the shares at or above HIGH."""
    names = {int(p["split"]): name for name, p in doc["parts"].items()}
    out = {}
    for split, ts in sorted(SH.by_split(src, [doc]).items()):
        m = ts.load(0, ts.tables, ("identity", "group"))
        ident, grp, n = m["identity"], np.asarray(m["group"]).astype(np.int64), ts.tables
        near = SC.nearest_identity(ident)
        fine, coarse = SC.fine_stratum(ident), SC.coarse_stratum(ident)
        has = ~np.isnan(near)
        top = lambda x, sel: dict(tables=int((near[sel] >= x - EPS).sum()), share=float((near[sel] >= x - EPS).mean()) if sel.any() else None,
                                  share_of_tables_with_rows=float((near[sel & has] >= x - EPS).mean()) if (sel & has).any() else None)
        every = np.ones(n, dtype=bool)
        d = dict(tables=n, with_other_rows=int(has.sum()), fine=_counts(fine, SC.FINE, n), coarse=_counts(coarse, SC.COARSE, n),
                 high={"from_%.2f" % x: top(x, every) for x in HIGH}, per_group={})
        for g, name in enumerate(groups):
            sel = grp == g
            if sel.any():
                d["per_group"][name] = dict(tables=int(sel.sum()), fine=_counts(fine[sel], SC.FINE, int(sel.sum())),
                                            high={"from_%.2f" % x: top(x, sel) for x in HIGH})
        out[names[split]] = d
    return out


def cap_counts(cfg, src, doc, processes=2) -> dict:
    """Per split and group of the genomes of the table set's index (the open genomes): genes, genes whose homolog rows
    in stage pairs reach the stage's cap, and homolog rows per gene, from the source run of the tables."""
    import yaml
    from ci.stages.rel import w2_v0 as W
    from seeddrift.train.rel.tables import v0_build as V
    with open(os.path.join(src, "config.resolved.yaml")) as f:
        source = (yaml.safe_load(f).get("rel_w2_v0") or {}).get("source")
    if not source:
        raise SystemExit("rel_w2_census: the run of the tables names no source run (rel_w2_v0.source)")
    s = W.source(source)
    if s["provenance"] != doc["source"]:
        raise SystemExit("rel_w2_census: the source run %s is not the one the table set was built from" % source)
    ix = [f for f in doc["files"] if f.get("role") == "index"]
    if len(ix) != 1 or sha256_file(os.path.join(src, ix[0]["path"]), use_sidecar=False) != ix[0]["sha256"]:
        raise SystemExit("rel_w2_census: the index of the table set is missing or not the file its manifest names")
    with open(os.path.join(src, ix[0]["path"])) as f:
        genomes = json.load(f)["genomes"]
    if any(g["split"] not in V.OPEN_SPLITS for g in genomes):
        raise SystemExit("rel_w2_census: the index lists genomes outside the open splits")
    common.sealed_check(cfg, [dict(id=g["id"], taxid=int(g["taxid"])) for g in genomes], splits={g["id"]: g["split"] for g in genomes})
    paths = [os.path.join("data", "pairs", g["id"] + ".npz") for g in genomes]
    bad = [p for p, h in zip(paths, V.hash_files([os.path.join(source, p) for p in paths], processes)) if s["files"].get(p) != h]
    if bad:
        raise SystemExit("rel_w2_census: homolog files of %s differ from its manifests: %s" % (source, bad[:5]))
    limit = int((s["provenance"].get("pairs_settings") or {}).get("genomes") or 64)
    out = {}
    for g in genomes:
        with np.load(os.path.join(source, "data", "pairs", g["id"] + ".npz")) as z:
            rows = np.diff(z["off"].astype(np.int64))
        for key in (g["group"], "all"):
            d = out.setdefault(g["split"], {}).setdefault(key, dict(genomes=0, genes=0, genes_at_cap=0, rows=0))
            d["genomes"] += 1
            d["genes"] += int(len(rows))
            d["genes_at_cap"] += int((rows >= limit).sum())
            d["rows"] += int(rows.sum())
    for by in out.values():
        for d in by.values():
            d["share_at_cap"] = d["genes_at_cap"] / d["genes"] if d["genes"] else None
            d["rows_per_gene"] = d["rows"] / d["genes"] if d["genes"] else None
    train = (out.get("train") or {}).get("all", {}).get("genomes", 0)
    return dict(cap=limit, training_genomes=train, random_fill_ceiling=limit / max(train - 1, 1), per_split=out,
                note="genes of the open genomes whose homolog rows in stage pairs number the cap (the `genomes` nearest genomes kept per gene)")


def report(doc, counts, src=None, cap=None) -> dict:
    groups, rows = counts["groups"], int(counts["rows"])
    sets = {}
    for part, arrays in counts["parts"].items():
        sets[part] = dict(tables=doc["parts"][part]["tables"], sha256=doc["parts"][part]["sha256"], census=C.summarise(C.Census.from_arrays(arrays, rows), groups))
    corpus = C.corpus_counts(counts["genomes"], groups)
    out = dict(format="rel-w2-census", version=1, note="What the stored data gives the tables of version 0, counted while they were built. Development diagnosis.",
               source_run=doc["source"]["run_id"], tableset=dict(name=doc["name"], sha256=doc["sha256"], run_id=doc["producer"]["run_id"]),
               coverage="the %d groups of the medium corpus, train and held-out splits; far relatives only (rows among the 64 nearest genomes of a gene)" % len(groups),
               groups=groups, rows=rows, max_codons=counts.get("max_codons") or 0, corpus=corpus, sets=sets, caveats=CAVEATS)
    head = dict(codons_per_group={g: (corpus.get("train", {}).get(g) or {}).get("codons", 0) for g in groups},
                codons_train=(corpus.get("train", {}).get("all") or {}).get("codons", 0), codons_heldout=(corpus.get("heldout", {}).get("all") or {}).get("codons", 0))
    for part, d in sets.items():
        a = d["census"]["all"]
        near = a["nearest_row_identity"]["target"]
        head[part] = dict(
            tables=d["tables"], other_rows_mean=a["other_rows"]["mean"], tables_without_other_rows=a["other_rows"]["share_of_tables_without"],
            tables_without_other_rows_per_group={g: x["other_rows"]["share_of_tables_without"] for g, x in d["census"]["per_group"].items()},
            filled=dict(not_target=a["filled"]["not_target"]["share"], relation=a["filled"]["relation"]["share"], left=a["filled"]["left"]["share"],
                        right=a["filled"]["right"]["share"], not_target_of_row_slots=a["filled"]["not_target"]["share_of_row_slots"]),
            filled_per_group={g: x["filled"]["not_target"]["share"] for g, x in d["census"]["per_group"].items()},
            nearest_row_on_target=dict(quantiles=near["quantiles"], from_0_50=near["share_from_0_50"], from_0_90=near["share_from_0_90"],
                                       from_0_95=near["share_from_0_95"], from_0_99=near["share_from_0_99"], identical=near["share_identical"],
                                       without_any_row=near["share_without_any_row"]),
            nearest_row_on_relation=dict(quantiles=a["nearest_row_identity"]["relation"]["quantiles"], from_0_90=a["nearest_row_identity"]["relation"]["share_from_0_90"],
                                         without_any_row=a["nearest_row_identity"]["relation"]["share_without_any_row"]),
            random_rows_filled=dict(not_target=a["random_rows_filled"]["not_target"]["share"], target=a["random_rows_filled"]["target"]["share"]),
            random_rows_filled_per_group={g: x["random_rows_filled"]["not_target"]["share"] for g, x in d["census"]["per_group"].items()})
    if cap is not None:
        out["cap"] = cap
        out["caveats"] = CAVEATS + ["The fill of rows drawn at random is bounded by the cap of stage pairs: it keeps at most %d genomes per gene, "
                                    "counted over all three splits, so at most %d of the %d other training genomes (%.1f%%) can have a homolog row on "
                                    "a gene; that share measures the cap, not biology." % (
                                        cap["cap"], cap["cap"], cap["training_genomes"] - 1, 100 * cap["random_fill_ceiling"])]
        for split, by in cap["per_split"].items():
            head.setdefault(split, {})["genes_at_cap"] = dict(all=by["all"]["share_at_cap"], per_group={g: d["share_at_cap"] for g, d in by.items() if g != "all"})
    if src is not None:
        out["strata"] = strata(src, doc, groups)
        out["strata_note"] = ("nearest other row of every table at the target locus: the largest identity of rows 1.. "
                              "(meta field identity), in the strata of seeddrift/train/rel/evalh/score.py version %s" % SC.VERSION)
        for part, d in out["strata"].items():
            head.setdefault(part, {})
            head[part]["nearest_fine"] = {lab: x["share"] for lab, x in d["fine"].items()}
            head[part]["nearest_from_0_98"] = d["high"]["from_0.98"]
    out["headline"] = head
    return out


def build(st, cfg, run_dir):
    src, doc, counts = tables(run_dir, cfg)
    cap = cap_counts(cfg, src, doc, int((cfg.get("rel_w2_census") or {}).get("processes", 2)))
    rep = report(doc, counts, src, cap)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    page = os.path.join(folder, "report.md")
    with open(page + ".part", "w") as f:
        f.write(C.page(rep))
    os.replace(page + ".part", page)
    st.output("census_page", page)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("census_report", p)
    st.extra.update(tableset_sha256=doc["sha256"], tables_run=os.path.basename(src), headline=rep["headline"])
    return rep


def _shares(x, path=""):
    """Every value under a key that names a share, with its path."""
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and ("share" in k or k in ("not_target", "relation", "left", "right", "target")):
                yield path + "/" + k, v
            else:
                yield from _shares(v, path + "/" + k)


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        rep = json.load(f)
    _, doc, counts = tables(run_dir, cfg)
    problems = []
    if (rep.get("tableset") or {}).get("sha256") != doc["sha256"]:
        problems.append(dict(error="the report is not of this table set"))
    for part, d in doc["parts"].items():
        got = ((rep.get("sets") or {}).get(part) or {}).get("census", {}).get("all", {}).get("tables")
        if got != d["tables"]:
            problems.append(dict(part=part, error="the census counted %s tables, the table set has %d" % (got, d["tables"])))
        per = ((rep.get("sets") or {}).get(part) or {}).get("census", {}).get("per_group", {})
        want = [g for g in rep.get("groups") or [] if g not in d.get("groups_without_a_genome", [])]
        if sorted(per) != sorted(want):
            problems.append(dict(part=part, error="groups of the census differ from the groups of the table set"))
    bad = [k for k, v in _shares(rep.get("sets")) if not (isinstance(v, (int, float)) and math.isfinite(v) and -1e-9 <= v <= 1 + 1e-9)]
    if bad:
        problems.append(dict(error="values that are not shares", where=bad[:10]))
    for g in rep.get("groups") or []:
        if not ((rep.get("corpus") or {}).get("train", {}).get(g) or {}).get("codons"):
            problems.append(dict(group=g, error="no codons of training genomes"))
    for part, d in doc["parts"].items():
        x = (rep.get("strata") or {}).get(part) or {}
        for key in ("fine", "coarse"):
            if sum(v["tables"] for v in (x.get(key) or {}).values()) != d["tables"]:
                problems.append(dict(part=part, strata=key, error="the strata do not hold every table of the set"))
        if (x.get("high") or {}).get("from_0.98", {}).get("share") is None:
            problems.append(dict(part=part, error="no share of nearest rows at or above 0.98"))
    cap = rep.get("cap") or {}
    for split in {g.get("split") for g in (doc.get("genomes") or [])}:
        d = ((cap.get("per_split") or {}).get(split) or {}).get("all") or {}
        if not d.get("genes") or d.get("share_at_cap") is None or not 0 <= d["share_at_cap"] <= 1:
            problems.append(dict(split=split, error="no share of genes at the cap of stage pairs"))
    bad = [k for k, v in _shares(rep.get("strata")) if v is not None and not (math.isfinite(v) and -1e-9 <= v <= 1 + 1e-9)]
    if bad:
        problems.append(dict(error="strata values that are not shares", where=bad[:10]))
    page = os.path.join(run_dir, OUT, "report.md")
    if not os.path.exists(page) or os.path.getsize(page) < 200:
        problems.append(dict(error="the page is missing"))
    require(GateResult("G4_rel_w2_census", "fail" if problems else "pass", dict(problems=problems[:50], headline=rep.get("headline"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False), doc["sha256"]])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(run_dir, OUT, "report.md")) as f:
        md = f.read().splitlines()
    return dict(json=rep["headline"], md=md[2:])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w2_census", requires=() if (st_tables_from(run_dir)) else ("rel_w2_v0",)) as st:
        build(st, st.config, run_dir)


def st_tables_from(run_dir):
    import yaml
    with open(os.path.join(run_dir, "config.resolved.yaml")) as f:
        return bool((yaml.safe_load(f).get("rel_w2_census") or {}).get("tables_from"))


if __name__ == "__main__":
    main()
