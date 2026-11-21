"""Stages of pipeline family_a_eval (benchmark v2 family A, bench/hard/V2_FAMILY_A.md part 3), on the labels of a passed
family_a run (`famAe.labels_from`), which they never change:

  units  element clusters of the insertion segments and the merge-rule units, from the labels and the development
         records alone (no score is read): data/famA/units.json
  eval   the arms (main: insertion against 1:1 matched core; core; noise; v1, the old island set reproduced), the
         matching ladder, AUROCs per chromosome, unit and overall with the protocol's paired cluster-bootstrap
         intervals, the contrasts, the published strata and the controls (balance, own covariates, shuffle,
         reproduction, lk_og): data/famA/eval/{report.json, arms.tsv, balance.tsv, ladder.tsv}

Scores of record: the audit run's columns (cmp_*, tv_*) and the island runs' model columns (`<label>:<column>`);
covariates and lk_og from the first island run's hits after the V1 view.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys

import numpy as np
import yaml

from ci.hashing import sha256_file
from ci.stages import family_a as FA
from ci.stages.common import Stage, main_args, write_json

ARMS = ("main", "core", "noise", "v1")
COMPARE = ("cmp_codon", "cmp_kmer", "cmp_hits", "tv_win_dev")
COVS = ("cov_len", "cov_hom", "cov_ident")
CATS = ("mobile", "ribosomal", "transport", "regulator", "hypothetical", "other")       # priority for the exact field
NO_HITS = ("cmp_codon", "cmp_gc3", "cmp_kmer", "cmp_short")              # with tv_*: the scores that read no hit


def _tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _floats(rows, key):
    return np.array([float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows], float)


def order_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


# --- inputs ----------------------------------------------------------------------------------------------------------

def label_source(src):
    """The checked outputs of a passed family_a run: its labels and presence indexes and the label tables' sha256."""
    from ci.stages.codon import gate_passed, stage_output
    base = os.path.join(src, "data", "famA")
    out = dict(run=src, labels_index=stage_output(src, "famA_labels", "famA_labels", os.path.join(base, "labels", "index.json")),
               presence_index=stage_output(src, "famA_presence", "famA_presence", os.path.join(base, "presence", "index.json")))
    gate_passed(src, "G4_famA_labels")
    with open(os.path.join(base, "labels", "index.json")) as f:
        idx = json.load(f)
    out["tables"] = {a: sha256_file(os.path.join(base, "labels", a + ".tsv"), use_sidecar=False) for a in sorted(idx["counts"])}
    return out, idx


def island_source(run, accs, first):
    """The checked outputs of a passed island run: its index, gate and scores record; for the first run also the hits
    and genes.tsv of every chromosome (manifest-checked) and the reference library. Model score tables are not in a
    manifest; their sha256 are recorded."""
    from ci.stages.codon import check_files, gate_passed, stage_output
    ip = os.path.join(run, "data", "islands", "index.json")
    rec = dict(run=run, islands_index=stage_output(run, "islands_data", "islands_index", ip),
               scores=stage_output(run, "islands_score", "islands_scores", os.path.join(run, "data", "islands", "scores.json")))
    gate_passed(run, "G3_islands")
    with open(ip) as f:
        idx = json.load(f)
    folders = {g["id"]: g["folder"] for g in idx["genomes"]}
    missing = [a for a in accs if a not in folders]
    if missing:
        raise SystemExit("family_a_eval: island run %s lacks %s" % (run, missing[:5]))
    with open(os.path.join(run, "record.json")) as f:
        rec["taxdump_sha256"] = json.load(f).get("taxdump_sha256")
    if first:
        mp = os.path.join(run, "data", "islands", "manifest.json")
        stage_output(run, "islands_data", "islands_manifest", mp)
        with open(mp) as f:
            man = json.load(f)
        msha = man["sha256"]
        rec["views"] = (man.get("meta") or {}).get("views")
        if not rec["views"]:
            raise SystemExit("family_a_eval: the hits of %s were not searched under a view" % run)
        check_files(run, os.path.join(run, "manifests", msha + ".json"),
                    ["data/islands/hits/%s.npz" % a for a in accs] + ["%s/genes.tsv" % folders[a] for a in accs])
        rec["library"] = stage_output(run, "reference", "library", os.path.join(run, "data", "reference", "library.npz"))
    with open(os.path.join(run, "data", "islands", "scores.json")) as f:
        sj = json.load(f)
    rec["variants"] = list(sj.get("variants") or [])
    rec["tables"] = {}
    for v in rec["variants"]:
        for a in accs:
            p = os.path.join(run, "data", "islands", "scores", v, "scores_%s.tsv" % a)
            rec["tables"]["%s/%s" % (v, a)] = sha256_file(p, use_sidecar=False)
    return rec, folders


def audit_source(run):
    """The checked outputs of a passed audit run: its columns record, report and gate."""
    from ci.stages.codon import gate_passed, stage_output
    rec = dict(run=run, columns=stage_output(run, "audit_columns", "audit_columns", os.path.join(run, "data", "audit", "columns.json")),
               report=stage_output(run, "audit_report", "audit_report", os.path.join(run, "data", "audit", "report.json")))
    gate_passed(run, "G4_audit")
    return rec


# --- per-gene facts from the hits ------------------------------------------------------------------------------------

def hit_covariates(h):
    """(rows, best identity) per gene from an island hits npz (after the view)."""
    off = np.asarray(h["off"], np.int64)
    n = np.diff(off)
    pid = np.asarray(h["pid"], float) / 10.0
    best = np.zeros(len(n))
    for i in np.where(n > 0)[0]:
        best[i] = pid[off[i]:off[i + 1]].max()
    return n, best


def outgroup_leak(h, genus_rank, genera_ids):
    """lk_og per gene: the number of outgroup genera (each a set of NCBI genus taxids) with a visible hit row, and the
    best identity of those rows (0 without one). genus_rank: the library's genus taxid per proteome."""
    off = np.asarray(h["off"], np.int64)
    prot = np.asarray(h["prot"], np.int64)
    pid = np.asarray(h["pid"], float) / 10.0
    g = genus_rank[prot] if len(prot) else np.zeros(0, np.int64)
    n = len(off) - 1
    count, best = np.zeros(n), np.zeros(n)
    for ids in genera_ids:
        hit = np.isin(g, list(ids)) if ids else np.zeros(len(g), bool)
        for i in range(n):
            sl = slice(off[i], off[i + 1])
            m = hit[sl]
            if m.any():
                count[i] += 1
                best[i] = max(best[i], pid[sl][m].max())
    return count, best


# --- matching and metrics --------------------------------------------------------------------------------------------

def quantiles(v, idx):
    """Within-chromosome quantiles of v over the genes idx (average ranks / (n - 1)); NaN where v is missing."""
    from ci.stages import protocol as PR
    out = np.full(len(v), np.nan)
    sub = np.asarray(v, float)[idx]
    out[idx] = PR.within_quantile(sub)
    return out


def match_pairs(acc, names, pos, neg, Q, exact, caliper):
    """1:1 matching without replacement (V2_FAMILY_A.md part 3): positives in ascending SHA-256 of
    "<chromosome>|<gene>|match-v1" each take the unused negative with equal exact fields, every |quantile difference| at
    most caliper and the smallest sum of differences (ties by the same key). Q: [genes, k] quantiles (a NaN never
    matches); exact: [genes, m]. Returns {positive index: negative index}."""
    key = {i: order_key("%s|%s|match-v1" % (acc, names[i])) for i in list(pos) + list(neg)}
    neg = np.array(sorted(neg, key=lambda i: key[i]), int)
    used = np.zeros(len(neg), bool)
    out = {}
    for p in sorted(pos, key=lambda i: key[i]):
        if not len(neg):
            break
        d = np.abs(Q[neg] - Q[p])
        ok = ~used & np.all(d <= caliper, 1) & np.all(exact[neg] == exact[p], 1) & np.all(np.isfinite(d), 1)
        if not ok.any():
            continue
        cost = np.where(ok, np.round(d.sum(1), 9), np.inf)     # rounded: equal sums tie and the key order decides
        j = int(np.argmin(cost))                          # the first minimum: negatives are in key order
        used[j] = True
        out[int(p)] = int(neg[j])
    return out


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if sd > 0:
        return float((a.mean() - b.mean()) / sd)
    return 0.0 if a.mean() == b.mean() else float("inf")


def arm_auroc(table, units, score):
    """(overall, {unit: mean}, {chromosome: AUROC}) of one score on one arm table: per chromosome, mean over the
    chromosomes of a unit, mean over units."""
    from ci.stages import protocol as PR
    per, byu = {}, {}
    for acc, t in sorted(table.items()):
        if t["unit"] not in units:
            continue
        v = np.asarray(t["cols"][score], float)
        a = PR.auroc(v[t["y"] == 1], v[t["y"] == 0])
        if a is not None:
            per[acc] = a
            byu.setdefault(t["unit"], []).append(a)
    um = {u: float(np.mean(x)) for u, x in sorted(byu.items())}
    return (float(np.mean(list(um.values()))) if um else None), um, per


def dominant_category(meta_row):
    for c in CATS:
        v = meta_row.get("md_cat_" + c)
        if v not in (None, "") and float(v) > 0:
            return c
    return "other"


# --- stages ----------------------------------------------------------------------------------------------------------

def _spec(st, fe):
    with open(st.repo(fe.get("spec", FA.SPEC))) as f:
        return yaml.safe_load(f)


def do_units(run_dir):
    with Stage(run_dir, "famAe_units", requires=("pre",)) as st:
        cfg = st.config
        fe = cfg["famAe"]
        spec = _spec(st, fe)
        src, lidx = label_source(fe["labels_from"])
        base = os.path.join(fe["labels_from"], "data", "famA", "labels")
        segs, sizes, proteins, genus = {}, {}, {}, {}
        with open(st.repo("bench/islands/results/dev_genomes.tsv")) as f:
            rsha = {r["accession"]: r["sha256"] for r in csv.DictReader(f, delimiter="\t")}
        for acc in sorted(lidx["counts"]):
            tab = FA.read_label_table(os.path.join(base, acc + ".tsv"))
            gb = os.path.join(fe["store"], "dev_genomes", acc + ".gb")
            if sha256_file(gb, use_sidecar=False) != rsha.get(acc):
                raise SystemExit("family_a_eval: the record of %s is not the one dev_genomes.tsv names" % acc)
            g = FA.record_genes(gb)
            if g["names"] != tab["names"]:
                raise SystemExit("family_a_eval: the label genes of %s are not its record's" % acc)
            for nm, p in zip(g["names"], g["proteins"]):
                proteins[(acc, nm)] = p
            for k, idx in FA.insertion_segments(tab).items():
                segs[(acc, k)] = [tab["names"][i] for i in idx]
                sizes[(acc, k)] = int((tab["run"] == k).sum())          # the whole S1 run (the threshold's denominator)
            genus[acc] = lidx["unit"][acc]
        clusters, links, n_hits = FA.element_clusters(segs, proteins, cfg["tools"]["diamond"], int(fe.get("threads", 8)),
                                                      os.path.join(run_dir, "tmp", "famAe"), spec["units"], sizes)
        units, cross = FA.merge_units(clusters, genus, set(genus.values()))
        unit_of = {acc: next(u for u, gs in units.items() if genus[acc] in gs) for acc in genus}
        p = st.path("famA", "units.json")
        write_json(p, dict(rule="merge", source=src, thresholds=spec["units"], segments=len(segs), hits=n_hits,
                           links=[dict(a="%s:%d" % a, b="%s:%d" % b, genes=n) for a, b, n in links],
                           clusters=[dict(id=cid, segments=["%s:%d" % x for x in g], chromosomes=sorted({c for c, _ in g})) for cid, g in clusters],
                           cross_genus=cross, units=units, chromosome_unit=unit_of, genus=genus))
        st.output("famAe_units", p)
        st.extra.update(units=sorted(units), clusters=len(clusters), cross_genus=len(cross), segments=len(segs))


def build_tables(fe, spec, lidx, units, isl, folders0, tax):
    """Per chromosome: names, label classes, published, coordinates, the score columns, covariates, lk_og and the
    category; the gene lists of the labels, the island runs and the audit must agree."""
    base = os.path.join(fe["labels_from"], "data", "famA")
    aud = fe["audit_from"]
    lib = np.load(os.path.join(isl[0]["run"], "data", "reference", "library.npz"))
    genus_rank = np.asarray(lib["ranks"], np.int64)[:, 4]
    with open(os.path.join(base, "presence", "index.json")) as f:
        pidx = json.load(f)
    out = {}
    for acc in sorted(lidx["counts"]):
        tab = FA.read_label_table(os.path.join(base, "labels", acc + ".tsv"))
        names = tab["names"]
        genes = _tsv(os.path.join(isl[0]["run"], folders0[acc], "genes.tsv"))
        if [r["gene_id"] for r in genes] != names:
            raise SystemExit("family_a_eval: the label genes of %s differ from the island run's genes.tsv" % acc)
        cols = {}
        arows = _tsv(os.path.join(aud, "data", "audit", "columns", "scores_%s.tsv" % acc))
        if [r["gene_id"] for r in arows] != names:
            raise SystemExit("family_a_eval: the label genes of %s differ from the audit's columns" % acc)
        for k in arows[0]:
            if k != "gene_id":
                cols[k] = _floats(arows, k)
        meta = _tsv(os.path.join(aud, "data", "audit", "columns", "meta_%s.tsv" % acc))
        if [r["gene_id"] for r in meta] != names:
            raise SystemExit("family_a_eval: the label genes of %s differ from the audit's meta" % acc)
        for r in isl:
            for v in r["variants"]:
                mrows = _tsv(os.path.join(r["run"], "data", "islands", "scores", v, "scores_%s.tsv" % acc))
                if [x["gene_id"] for x in mrows] != names:
                    raise SystemExit("family_a_eval: the genes of %s in %s/%s differ from the labels" % (acc, r["label"], v))
                for k in mrows[0]:
                    if k != "gene_id":
                        cols["%s:%s" % (r["label"], k)] = _floats(mrows, k)
        h = np.load(os.path.join(isl[0]["run"], "data", "islands", "hits", acc + ".npz"))
        rows, ident = hit_covariates(h)
        if len(rows) != len(names):
            raise SystemExit("family_a_eval: the hits of %s do not have one row range per gene" % acc)
        length = np.log(np.maximum((tab["end"] - tab["start"] + 1) / 3.0, 1.0))
        cols["cov_len"], cols["cov_hom"], cols["cov_ident"] = length, np.log1p(rows), ident
        info = pidx["chromosomes"][acc]
        og = sorted({d["genus"] for d in info["genomes"].values() if d["role"] == "outgroup" and d.get("genus")})
        ids = [set(tax.genus_by_name(re.sub(r"_[A-Z]+$", "", g))) for g in og]
        cols["lk_og"], cols["lk_og_id"] = outgroup_leak(h, genus_rank, ids)
        out[acc] = dict(names=names, cls=tab["cls"], published=tab["published"], start=tab["start"], cols=cols, has_hit=(rows > 0).astype(float),
                        cat=np.array([dominant_category(m) for m in meta], dtype=object), unit=units[acc], outgroups=og)
    return out


def arm_tables(T, spec, ladder_only=False):
    """The main, core and noise arm tables of the evaluated chromosomes, the matching record and the ladder."""
    ec = spec["eval"]
    main, core, noise, record, ladder = {}, {}, {}, {}, {}
    steps = [("len", ["cov_len"], []), ("len+hom", ["cov_len", "cov_hom"], []), ("len+hom+ident", list(COVS), []),
             ("+cai", list(COVS) + ["tv_cai"], []), ("+neighbourhood", list(COVS) + ["tv_cai", "tv_win_gc", "tv_win_dev"], []),
             ("+category", list(COVS) + ["tv_cai", "tv_win_gc", "tv_win_dev"], ["cat"])]
    for acc, t in sorted(T.items()):
        pos = np.where(t["cls"] == "insertion")[0]
        neg = np.where(t["cls"] == "core")[0]
        if len(pos) < int(ec["min_pos"]) or len(neg) < int(ec["min_neg"]):
            record[acc] = dict(evaluated=False, positives=int(len(pos)), core=int(len(neg)))
            continue
        lab = np.concatenate([pos, neg])
        for name, covs, ex in steps:
            Q = np.stack([quantiles(t["cols"][c], lab) for c in covs], 1)
            exact = np.stack([t["has_hit"]] + [np.array([CATS.index(x) for x in t["cat"]], float) for _ in ex], 1)
            m = match_pairs(acc, t["names"], pos, neg, Q, exact, float(ec["caliper"]))
            by = {}
            for p in pos:
                by.setdefault(t["cat"][p], [0, 0])
                by[t["cat"][p]][0] += 1
                by[t["cat"][p]][1] += int(p in m)
            ladder.setdefault(name, {})[acc] = dict(kept=len(m), positives=int(len(pos)), by_category=by)
            if name == "len+hom+ident":
                mm, qm = m, Q
        y = np.full(len(t["names"]), -1)
        y[pos], y[neg] = 1, 0
        core[acc] = dict(y=y, cols=t["cols"], unit=t["unit"], start=t["start"])
        y2 = np.full(len(t["names"]), -1)
        y2[list(mm)] = 1
        y2[list(mm.values())] = 0
        main[acc] = dict(y=y2, cols=t["cols"], unit=t["unit"], start=t["start"], q=qm, pairs=mm)
        y3 = np.full(len(t["names"]), -1)
        y3[pos] = 1
        y3[t["cls"] == "accessory"] = 0
        noise[acc] = dict(y=y3, cols=t["cols"], unit=t["unit"], start=t["start"])
        record[acc] = dict(evaluated=True, positives=int(len(pos)), core=int(len(neg)), kept=len(mm), accessory=int((t["cls"] == "accessory").sum()))
    return dict(main=main, core=core, noise=noise), record, ladder


def v1_arm(fe, audit_report, T):
    """The old island set's gated labels on the audit's chromosomes and units, with every score of T (the audit's
    columns of these chromosomes were checked against the labels' gene lists)."""
    with open(os.path.join(fe["audit_from"], "config.resolved.yaml")) as f:
        acfg = yaml.safe_load(f)
    isl = acfg["audit"]["islands_from"]
    with open(os.path.join(isl, "data", "islands", "eval", "dev.json")) as f:
        units_of = json.load(f)["units"]
    with open(os.path.join(isl, "config.resolved.yaml")) as f:
        store = (acfg.get("islands") or {}).get("store") or yaml.safe_load(f)["islands"]["store"]
    out = {}
    for acc in audit_report["chromosomes"]:
        if acc not in T:
            raise SystemExit("family_a_eval: the audit's chromosome %s has no labels" % acc)
        lab = {r["gene_id"]: r["class"] for r in _tsv(os.path.join(store, "labels_panel", acc + ".tsv"))}
        y = np.array([1 if lab.get(n) == "island" else 0 if lab.get(n) == "native" else -1 for n in T[acc]["names"]])
        out[acc] = dict(y=y, cols=T[acc]["cols"], unit=units_of[acc], start=T[acc]["start"])
    return out


def do_eval(run_dir):
    with Stage(run_dir, "famAe_eval", requires=("pre",)) as st:
        from ci.stages import audit as AU
        from ci.stages import protocol as PR
        from ci.taxonomy import Taxonomy
        cfg = st.config
        fe = cfg["famAe"]
        spec = _spec(st, fe)
        ec = spec["eval"]
        src, lidx = label_source(fe["labels_from"])
        with open(st.path("famA", "units.json")) as f:
            uj = json.load(f)
        if uj["source"]["labels_index"] != src["labels_index"] or uj["source"]["tables"] != src["tables"]:
            raise SystemExit("family_a_eval: the units were computed on other labels")
        accs = sorted(lidx["counts"])
        isl, folders0 = [], None
        for k, r in enumerate(fe["islands_from"]):
            rec, folders = island_source(r["run"], accs, k == 0)
            rec["label"] = r["label"]
            isl.append(rec)
            folders0 = folders0 or folders
        aud = audit_source(fe["audit_from"])
        from ci.hashing import taxdump_hash
        tsha = taxdump_hash(cfg["taxdump"])
        if any(r.get("taxdump_sha256") != tsha for r in isl):
            raise SystemExit("family_a_eval: an island run used another taxdump than this run")
        with open(os.path.join(fe["audit_from"], "data", "audit", "report.json")) as f:
            areport = json.load(f)
        T = build_tables(fe, spec, lidx, uj["chromosome_unit"], isl, folders0, Taxonomy(st.repo(cfg["taxdump"])))
        tables, record, ladder = arm_tables(T, spec)
        tables["v1"] = v1_arm(fe, areport, T)
        units = sorted({t["unit"] for t in tables["core"].values()})
        audit_cols = [k for k in T[accs[0]]["cols"] if re.match(r"^(cmp|tv)_", k)]
        model_cols = sorted(k for k in T[accs[0]]["cols"] if ":" in k)
        scores = audit_cols + model_cols + list(COVS) + ["lk_og", "lk_og_id"]
        report = dict(sources=dict(labels=src, audit=aud, islands=isl), units=uj["units"], chromosome_unit=uj["chromosome_unit"],
                      chromosomes=record, scores=scores, arms={}, intervals={}, contrasts={}, strata={})
        boot, block, seed = int(ec["boot"]), int(ec["block"]), int(cfg.get("seed", 1))
        vunits = sorted({t["unit"] for t in tables["v1"].values()})
        for arm in ARMS:
            tb, us = tables[arm], (vunits if arm == "v1" else units)
            report["arms"][arm] = {}
            for s in scores:
                a, um, per = arm_auroc(tb, us, s)
                report["arms"][arm][s] = dict(auroc=a, per_unit=um, per_chromosome=per)
            tb = {a: dict(y=t["y"], cols=t["cols"], unit=t["unit"], start=t["start"]) for a, t in tb.items()}
            rows = PR.paired_delta(tb, us, [(s, None) for s in scores], n_boot=boot, seed=seed, block=block)
            report["intervals"][arm] = {r["a"]: dict(lo=r["lo"], hi=r["hi"], delta=r["delta"], n_boot=r["n_boot"]) for r in rows}
            if arm in ("main", "core"):
                report["contrasts"][arm] = PR.paired_delta(tb, us, [(m, c) for m in model_cols for c in COMPARE if c in scores], n_boot=boot, seed=seed,
                                                           block=block)
        # the published strata of the core arm
        for name, keep in (("published_island", "island"), ("not_published", None)):
            tb = {}
            for acc, t in tables["core"].items():
                y = t["y"].copy()
                pub = T[acc]["published"]
                drop = (y == 1) & ((pub != "island") if keep else (pub == "island"))
                y[drop] = -1
                tb[acc] = dict(y=y, cols=t["cols"], unit=t["unit"])
            report["strata"][name] = {s: arm_auroc(tb, units, s)[0] for s in scores}
        # v1: the audit's chromosomes and units; its columns compared with the audit's report
        for s in audit_cols:
            r = report["arms"]["v1"][s]
            ref = (areport.get("scores") or {}).get(s, {}).get("auroc")
            r.update(audit=ref, diff=None if r["auroc"] is None or ref is None else abs(r["auroc"] - ref))
        # controls
        shuffled = AU.shuffle_control({a: dict(y=t["y"], cols=t["cols"]) for a, t in tables["core"].items()}, scores,
                                      int(ec["shuffle_draws"]), seed)
        balance = []
        for u in units:
            ch = [a for a, t in tables["main"].items() if t["unit"] == u]
            for k, c in enumerate(COVS):
                before_p = np.concatenate([tables["main"][a]["q"][T[a]["cls"] == "insertion", k] for a in ch]) if ch else np.zeros(0)
                before_n = np.concatenate([tables["main"][a]["q"][T[a]["cls"] == "core", k] for a in ch]) if ch else np.zeros(0)
                qp = np.concatenate([tables["main"][a]["q"][list(tables["main"][a]["pairs"]), k] for a in ch]) if ch else np.zeros(0)
                qn = np.concatenate([tables["main"][a]["q"][list(tables["main"][a]["pairs"].values()), k] for a in ch]) if ch else np.zeros(0)
                balance.append(dict(unit=u, covariate=c, smd_before=smd(before_p, before_n), smd_after=smd(qp, qn), kept=int(len(qp))))
        lk = report["arms"]["main"]["lk_og"]["auroc"]
        flag = bool(lk is not None and lk > float(ec["lk_og_flag"]))
        hit_reading = [s for s in scores if not (s in NO_HITS or s.startswith("tv_"))]
        if flag:                                          # the hit-reading scores are marked
            for arm in report["arms"].values():
                for s in hit_reading:
                    arm[s]["label_evidence_visible"] = True
        report.update(shuffle=shuffled, balance=balance, ladder={k: dict(kept=sum(x["kept"] for x in v.values()), positives=sum(x["positives"] for x in v.values()),
                                                                            per_chromosome=v) for k, v in ladder.items()},
                      lk_og=dict(auroc_main=lk, flag=flag, hit_reading=hit_reading), evaluated_units=units,
                      thresholds=ec)
        out = st.path("famA", "eval")
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "report.json"), report)
        with open(os.path.join(out, "arms.tsv"), "w") as f:
            f.write("arm\tscore\tauroc\tlo\thi\tlabel_evidence_visible\tper_unit\n")
            for arm in ARMS:
                for s, r in (report["arms"].get(arm) or {}).items():
                    iv = (report["intervals"].get(arm) or {}).get(s, {})
                    f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (arm, s, r["auroc"], iv.get("lo", ""), iv.get("hi", ""), bool(r.get("label_evidence_visible")),
                                                       json.dumps(r.get("per_unit"), sort_keys=True)))
        with open(os.path.join(out, "balance.tsv"), "w") as f:
            f.write("unit\tcovariate\tsmd_before\tsmd_after\tkept\n")
            for b in balance:
                f.write("%s\t%s\t%s\t%s\t%d\n" % (b["unit"], b["covariate"], b["smd_before"], b["smd_after"], b["kept"]))
        with open(os.path.join(out, "ladder.tsv"), "w") as f:
            f.write("step\tkept\tpositives\tkept_share\n")
            for k, v in report["ladder"].items():
                f.write("%s\t%d\t%d\t%.4f\n" % (k, v["kept"], v["positives"], v["kept"] / max(v["positives"], 1)))
        st.output("famAe_eval", os.path.join(out, "report.json"))
        st.extra.update(units=units, chromosomes=sum(1 for r in record.values() if r["evaluated"]), scores=len(scores),
                        kept=report["ladder"]["len+hom+ident"]["kept"], positives=report["ladder"]["len+hom+ident"]["positives"])


def main():
    run_dir = main_args()
    {"units": do_units, "eval": do_eval}[sys.argv[2]](run_dir)


if __name__ == "__main__":
    main()
