"""Stage rel_w2_dist (package W2; pipeline rel_tables): distances between the genomes of the open splits from shared
amino-acid k-mers (seeddrift/train/rel/tables/v0_dist.py), their distribution per group, and how a set of provisional
distance tiers would be occupied. `python -m ci.stages.rel.w2_dist RUN_DIR`. For sampling and reporting only: no
distance, tier or score is ever a model input.

Inputs: the training arrays of a passed selfsup run (rel_w2_dist.source), each file checked against the checksums its
records name, of the train and held-out splits only (the sealed check runs first; the sealed test split is never
opened). With rel_w2_dist.tables_from (a passed run of stage rel_w2_v0 on the same source), the distances are set
against the rows of those tables: per tier, how many rows and how identical they are to row 0 at the target locus.

Configuration (profile block rel_w2_dist): source, expect ({split: genomes}), ks (k-mer lengths, the first one primary;
default [6]), size (sketch size, 4096), edges (provisional tier edges on the distance of the primary k), processes,
tables_from.
Outputs under data/rel_w2/dist/: genomes.json (the genome list of the matrices: id, group, split, taxid, distinct
k-mers per k), distances_k<k>.npy (float32 [G, G], the Mash distance, inf without a shared k-mer) and
containment_k<k>.npy for every k, report.json (written last) and report.md.
Gate G4_rel_w2_dist (gate()): every open genome has its row; the matrices are G x G, symmetric, zero on the diagonal;
the genomes are those of the open splits of the source; every file is the one the report names.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from seeddrift.train.rel import handoff
from seeddrift.train.rel.tables import v0_build as V
from seeddrift.train.rel.tables import v0_dist as D

OUT = os.path.join("data", "rel_w2", "dist")
KEYS = {"source", "expect", "ks", "size", "edges", "processes", "tables_from"}
BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5)          # report bins of the distance
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def settings(cfg) -> dict:
    c = cfg.get("rel_w2_dist") or {}
    unknown = set(c) - KEYS
    if unknown or not c.get("source"):
        raise SystemExit("rel_w2_dist: needs `source`; unknown keys %s" % sorted(unknown))
    out = dict(source=str(c["source"]), expect={str(k): int(v) for k, v in (c.get("expect") or {}).items()}, ks=[int(k) for k in c.get("ks") or [6]],
               size=int(c.get("size", 4096)), edges=[float(x) for x in c.get("edges") or [0.05, 0.2, 0.5, 1.0]], processes=int(c.get("processes", 4)),
               tables_from=c.get("tables_from"), max_left_out=20)
    if not (out["ks"] and all(3 <= k <= 12 for k in out["ks"]) and len(set(out["ks"])) == len(out["ks"]) and 64 <= out["size"] <= 1 << 16
            and out["edges"] == sorted(out["edges"]) and len(set(out["edges"])) == len(out["edges"]) and 1 <= out["processes"] <= 16):
        raise SystemExit("rel_w2_dist: settings out of range: %r" % out)
    return out


_G = {}


def _sketch(args):
    gid, path, ks, size = args
    with np.load(path) as z:
        codons, coff = z["codons"], z["codon_off"]
    return gid, [D.sketch(codons, coff, k, size) for k in ks]


def _quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {("q%02d" % round(100 * q)): (float(np.quantile(x, q)) if len(x) else None) for q in QS}


def _hist(x):
    x = np.asarray(x, dtype=np.float64)
    fin = x[np.isfinite(x)]
    edges = np.asarray(BINS + (np.inf,))
    c = np.histogram(fin, bins=edges)[0] if len(fin) else np.zeros(len(BINS), dtype=np.int64)
    out = {("%.2f-%.2f" % (BINS[i], BINS[i + 1]) if i + 1 < len(BINS) else ">=%.2f" % BINS[i]): int(c[i]) for i in range(len(BINS))}
    out["no shared k-mer"] = int((~np.isfinite(x)).sum())
    return out


def _ranks(x):
    """Ranks with ties averaged (inf ties with inf)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    first = np.concatenate([[True], xs[1:] != xs[:-1]]) if len(xs) else np.zeros(0, dtype=bool)
    idx = np.flatnonzero(first)
    counts = np.diff(np.concatenate([idx, [len(x)]]))
    r = np.empty(len(x))
    r[order] = np.repeat(idx + (counts - 1) / 2.0, counts)
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    sa, sb = ra.std(), rb.std()
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else None


def tier_names(edges):
    e = list(edges)
    return ["<%.2f" % e[0]] + ["%.2f-%.2f" % (e[i], e[i + 1]) for i in range(len(e) - 1)] + [">=%.2f" % e[-1], "no shared k-mer"]


def per_genome(dist, genomes, edges):
    """For every genome A: the distances to the training genomes other than A, the nearest of them, the tier counts."""
    train = np.array([g["split"] == "train" for g in genomes])
    names = tier_names(edges)
    out = []
    for a, g in enumerate(genomes):
        others = train.copy()
        others[a] = False
        d = dist[a, others]
        t = np.bincount(D.tier_of(d, edges), minlength=len(names))
        near = int(np.flatnonzero(others)[np.argmin(d)]) if len(d) and np.isfinite(d).any() else -1
        out.append(dict(d=d, nearest=float(d.min()) if len(d) else float("inf"), nearest_index=near, tiers=t))
    return out, names


def summarise(dist, genomes, edges, groups):
    """Per split and group of genome A (and all): the nearest training genome's distance, all distances to training
    genomes, and the occupancy of the provisional tiers."""
    rows, names = per_genome(dist, genomes, edges)
    out = {}
    for split in ("train", "heldout"):
        by = {}
        for name in groups + ["all"]:
            sel = [i for i, g in enumerate(genomes) if g["split"] == split and (name == "all" or g["group"] == name)]
            if not sel:
                continue
            near = np.array([rows[i]["nearest"] for i in sel])
            same = [genomes[rows[i]["nearest_index"]]["group"] == genomes[i]["group"] for i in sel if rows[i]["nearest_index"] >= 0]
            alld = np.concatenate([rows[i]["d"] for i in sel])
            tiers = np.stack([rows[i]["tiers"] for i in sel])
            by[name] = dict(genomes=len(sel), nearest=dict(quantiles=_quantiles(near), histogram=_hist(near), share_without_any=float((~np.isfinite(near)).mean()),
                                                           share_nearest_in_the_same_group=float(np.mean(same)) if same else None),
                            pairs=dict(quantiles=_quantiles(alld), histogram=_hist(alld), share_without_shared_kmer=float((~np.isfinite(alld)).mean())),
                            tiers=dict(names=names, training_genomes_per_genome={n: float(tiers[:, k].mean()) for k, n in enumerate(names)},
                                       share_of_genomes_with_one={n: float((tiers[:, k] > 0).mean()) for k, n in enumerate(names)},
                                       occupied_per_genome=float((tiers[:, :-1] > 0).sum(1).mean())))
        out[split] = by
    return out


def against_rows(cfg, s, dist, genomes, groups):
    """The rows of the tables of a passed rel_w2_v0 run: per tier of the distance between the row's genome and row 0's,
    how many rows and their identity to row 0 at the target locus; Spearman's rank correlation of the two."""
    from seeddrift.train.rel.evalh import shards as SH
    try:
        doc = handoff.from_run(s["tables_from"], os.path.join("data", "rel_w2", "v0", "tableset.json"))
    except handoff.HandoffError as err:
        raise SystemExit("rel_w2_dist: %s" % err)
    ix = [f for f in doc["files"] if f.get("role") == "index"]
    if len(ix) != 1 or sha256_file(os.path.join(s["tables_from"], ix[0]["path"]), use_sidecar=False) != ix[0]["sha256"]:
        raise SystemExit("rel_w2_dist: the index of the table set is missing or not the file its manifest names")
    with open(os.path.join(s["tables_from"], ix[0]["path"])) as f:
        listed = [g["id"] for g in json.load(f)["genomes"]]
    at = {g["id"]: i for i, g in enumerate(genomes)}
    missing = sorted(set(listed) - set(at))
    if missing:
        raise SystemExit("rel_w2_dist: genomes of the tables without a distance: %s" % missing[:5])
    m = np.array([at[g] for g in listed], dtype=np.int64)
    names = tier_names(s["edges"])
    out = dict(tables_run=os.path.basename(os.path.normpath(s["tables_from"])), tableset_sha256=doc["sha256"], tiers=names, parts={})
    for split, ts in sorted(SH.by_split(s["tables_from"], [doc]).items()):
        z = ts.load(0, ts.tables, ("genome", "identity", "group"))
        g0 = np.repeat(z["genome"][:, :1], z["genome"].shape[1] - 1, axis=1)
        gq, ident = z["genome"][:, 1:], z["identity"][:, 1:].astype(np.float64)
        ok = gq >= 0
        d = dist[m[g0[ok]], m[gq[ok]]].astype(np.float64)
        idn = ident[ok]
        grp = np.repeat(np.asarray(z["group"]).astype(np.int64)[:, None], gq.shape[1], axis=1)[ok]
        t = D.tier_of(d, s["edges"])
        tiers = {}
        for k, name in enumerate(names):
            sel = t == k
            tiers[name] = dict(rows=int(sel.sum()), share=float(sel.mean()) if len(t) else None,
                               identity=dict(mean=float(idn[sel].mean()) if sel.any() else None, quantiles=_quantiles(idn[sel]) if sel.any() else None))
        part = "train" if split == 0 else "heldout"
        out["parts"][part] = dict(rows=int(ok.sum()), tiers=tiers, spearman_distance_identity=spearman(d, idn),
                                  spearman_per_group={name: spearman(d[grp == gi], idn[grp == gi]) for gi, name in enumerate(groups)})
    return out


def build(st, cfg, run_dir):
    s, t0 = settings(cfg), time.time()
    say = lambda text: print("rel_w2_dist: " + text, flush=True)
    src = W.source(s["source"])
    sealed = common.sealed_set(cfg)
    metas, left, sizes = W.open_genomes(src, dict(s, expect=s["expect"]), sealed)
    from ci.taxonomy import Taxonomy
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    ents = [W.entry(src, m) for m in metas]
    st.extra["sealed_check"] = common.sealed_check(cfg, ents, splits={m["id"]: m["split"] for m in metas}, sealed=sealed, tax=tax)
    st.extra.update(source=src["provenance"], split_sizes=sizes, left_out=left)
    paths = [os.path.join("data", "train_data", m["id"] + ".npz") for m in metas]
    bad = [p for p, h in zip(paths, V.hash_files([os.path.join(src["run"], p) for p in paths], s["processes"])) if src["files"].get(p) != h]
    if bad:
        raise SystemExit("rel_w2_dist: files of %s differ from its manifests: %s" % (src["run"], bad[:5]))
    say("%d training-array files match the source's manifests, %.0f s" % (len(paths), time.time() - t0))
    todo = [(m["id"], os.path.join(src["run"], p), s["ks"], s["size"]) for m, p in zip(metas, paths)]
    import multiprocessing
    with multiprocessing.get_context("fork").Pool(s["processes"]) as pool:
        got = dict(pool.map(_sketch, todo, chunksize=4))
    say("%d genomes sketched, %.0f s" % (len(got), time.time() - t0))
    groups = sorted({m["group"] for m in metas})
    genomes = [dict(id=m["id"], group=m["group"], split=m["split"], taxid=int(m["taxid"]), kmers={str(k): got[m["id"]][i][1] for i, k in enumerate(s["ks"])})
               for m in metas]
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    files, by_k = [], {}
    gp = os.path.join(folder, "genomes.json")
    write_json(gp, dict(genomes=genomes, ks=s["ks"], size=s["size"]))
    files.append(gp)
    primary = None
    for i, k in enumerate(s["ks"]):
        jac, con = D.matrix([got[m["id"]][i][0] for m in metas], [got[m["id"]][i][1] for m in metas], s["size"], s["processes"])
        dist = D.mash(jac, k)
        np.fill_diagonal(dist, 0.0)
        for name, arr in (("distances", dist), ("containment", con)):
            p = os.path.join(folder, "%s_k%d.npy" % (name, k))
            np.save(p, arr.astype(np.float32))
            files.append(p)
        by_k[str(k)] = summarise(dist, genomes, s["edges"], groups)
        if primary is None:
            primary = dist
        say("k=%d: %d pairs compared, %.0f s" % (k, len(metas) * (len(metas) - 1) // 2, time.time() - t0))
    rep = dict(format="rel-w2-dist", version=1, note=__doc__.strip().splitlines()[0], source=src["provenance"], ks=s["ks"], primary_k=s["ks"][0],
               size=s["size"], edges=s["edges"], tiers=tier_names(s["edges"]), groups=groups,
               genomes=dict(n=len(genomes), per_split={sp: sum(1 for g in genomes if g["split"] == sp) for sp in V.OPEN_SPLITS},
                            kmers={str(k): {name: _quantiles([g["kmers"][str(k)] for g in genomes if g["group"] == name]) for name in groups} for k in s["ks"]}),
               by_k=by_k, left_out=[dict(group=x["group"], split=x["split"], reasons=x["reasons"]) for x in left],
               caveats=["For sampling and reporting only; never a model input.",
                        "Only the first codons of a gene up to the source's limit and, for a eukaryote, its first 1,500 genes are in the arrays.",
                        "The Mash distance assumes k-mers that differ independently; above about 0.5 it is mostly the absence of shared k-mers.",
                        "A genome without a training genome sharing a k-mer has no finite nearest distance (counted apart)."])
    if s["tables_from"]:
        rep["rows"] = against_rows(cfg, s, primary, genomes, groups)
        say("the rows of the tables of %s set against the distances, %.0f s" % (rep["rows"]["tables_run"], time.time() - t0))
    rep["files"] = [dict(path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p)) for p in files]
    page = os.path.join(folder, "report.md")
    with open(page + ".part", "w") as f:
        f.write(page_of(rep))
    os.replace(page + ".part", page)
    st.output("dist_page", page)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("dist_report", p)
    st.extra.update(genomes=len(genomes), ks=s["ks"], seconds=round(time.time() - t0, 1))
    return rep


def _pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def _f(x):
    return "n/a" if x is None else "%.2f" % x


def page_of(rep):
    k = str(rep["primary_k"])
    L = ["# Distances between genomes (shared amino-acid %d-mers), version 0" % rep["primary_k"], "", rep["note"], "",
         "Source run `%s`; %d genomes of the train and held-out splits; sketch size %d; k = %s (the first is primary). Provisional tiers on the "
         "distance: %s." % (rep["source"]["run_id"], rep["genomes"]["n"], rep["size"], ", ".join(str(x) for x in rep["ks"]), ", ".join(rep["tiers"])), ""]
    for kk in [str(x) for x in rep["ks"]]:
        L += ["## k = %s: the nearest training genome of every genome" % kk, "",
              "| split | group | genomes | nearest q10 | q50 | q90 | without any | nearest in the same group | tiers occupied per genome |", "|---|---|---|---|---|---|---|---|---|"]
        for split, by in rep["by_k"][kk].items():
            for name, x in by.items():
                q = x["nearest"]["quantiles"]
                L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %.2f |" % (split, name, x["genomes"], _f(q["q10"]), _f(q["q50"]), _f(q["q90"]),
                                                                          _pct(x["nearest"]["share_without_any"]), _pct(x["nearest"]["share_nearest_in_the_same_group"]),
                                                                          x["tiers"]["occupied_per_genome"]))
        L.append("")
    a = rep["by_k"][k]["train"].get("all")
    if a:
        L += ["## Tiers (k = %s, training genomes as rows of a training genome)" % k, "", "| tier | training genomes per genome | share of genomes with at least one |",
              "|---|---|---|"]
        for n in rep["tiers"]:
            L.append("| %s | %.1f | %s |" % (n, a["tiers"]["training_genomes_per_genome"][n], _pct(a["tiers"]["share_of_genomes_with_one"][n])))
        L.append("")
    if rep.get("rows"):
        r = rep["rows"]
        L += ["## The rows of the tables of run `%s` by distance tier (k = %s)" % (r["tables_run"], k), ""]
        for part, x in r["parts"].items():
            L += ["%s tables: %s rows; Spearman's rank correlation of distance and identity to row 0: %s." % (
                part, "{:,}".format(x["rows"]), _f(x["spearman_distance_identity"])), "", "| tier | rows | share | identity mean | identity q50 |", "|---|---|---|---|---|"]
            for n, y in x["tiers"].items():
                L.append("| %s | %s | %s | %s | %s |" % (n, "{:,}".format(y["rows"]), _pct(y["share"]), _f(y["identity"]["mean"]),
                                                         _f((y["identity"]["quantiles"] or {}).get("q50"))))
            L.append("")
    L += ["## What these numbers are not", ""] + ["- " + x for x in rep["caveats"]] + [""]
    return "\n".join(L)


def gate(st, cfg, run_dir):
    s = settings(cfg)
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        rep = json.load(f)
    problems = []
    for e in rep.get("files") or []:
        full = os.path.join(run_dir, e["path"])
        if not os.path.exists(full) or sha256_file(full, use_sidecar=False) != e["sha256"]:
            problems.append(dict(file=e["path"], error="missing or not the file the report names"))
    with open(os.path.join(run_dir, OUT, "genomes.json")) as f:
        genomes = json.load(f)["genomes"]
    src = W.source(s["source"])
    want = sorted(m["id"] for m in src["index"]["genomes"] if m.get("source", "corpus") == "corpus" and m["split"] in V.OPEN_SPLITS)
    left = len(rep.get("left_out") or [])
    have = sorted(g["id"] for g in genomes)
    if any(g["split"] not in V.OPEN_SPLITS for g in genomes) or not set(have) <= set(want) or len(want) - len(have) != left:
        problems.append(dict(error="the genomes are not those of the open splits of the source (less the ones left out)"))
    for k in s["ks"]:
        dist = np.load(os.path.join(run_dir, OUT, "distances_k%d.npy" % k))
        if dist.shape != (len(genomes), len(genomes)) or not np.array_equal(dist, dist.T) or (np.diag(dist) != 0).any() or (dist < 0).any():
            problems.append(dict(k=k, error="the distance matrix is not square, symmetric, non-negative and zero on the diagonal"))
        if str(k) not in (rep.get("by_k") or {}):
            problems.append(dict(k=k, error="no summary"))
    if s["tables_from"] and not (rep.get("rows") or {}).get("parts"):
        problems.append(dict(error="tables_from is configured but the rows were not set against the distances"))
    require(GateResult("G4_rel_w2_dist", "fail" if problems else "pass", dict(problems=problems[:50], genomes=len(genomes), ks=s["ks"])), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    k = str(rep["primary_k"])
    js = dict(genomes=rep["genomes"]["n"], ks=rep["ks"], tiers=rep["tiers"],
              nearest_q50={sp: {g: x["nearest"]["quantiles"]["q50"] for g, x in by.items()} for sp, by in rep["by_k"][k].items()},
              occupied_per_genome={sp: by.get("all", {}).get("tiers", {}).get("occupied_per_genome") for sp, by in rep["by_k"][k].items()},
              spearman={part: x["spearman_distance_identity"] for part, x in (rep.get("rows") or {}).get("parts", {}).items()})
    return dict(json=js, md=page_of(rep).splitlines()[2:60])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w2_dist") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
