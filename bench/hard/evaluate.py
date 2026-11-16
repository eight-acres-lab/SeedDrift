#!/usr/bin/env python
"""Evaluator of the first freeze (PROTOCOL.md): island genes against positional-core natives.

  per chromosome   for every score and regime: R@1% and its companions at gene level (own score, and the window of 11
                   genes over scored genes), island regions found at block level, the label-shift null, calls by class
  label sets       gated (primary), pub (S-pub: published labels as they are), absent (S-a), deep (S-f)
  units            the genus (several chromosomes of a unit are averaged; --merge joins genera into one unit)
  contrasts        P1 and P2 as registered, with t-intervals over units, exact sign-flip tests, Holm inside an
                   endpoint, set contrasts, and the conditions under which a difference counts

Scores are read by column name from folders with scores_<accession>.tsv; the column names are the method names of the
protocol (codon, kmer, gc3, short, wide, wide_rule_ai, ai_all, ai_narrow, ai_gene, nohit, n_rows, absent_near,
best_id, sim_anom, copies, viral, random). fuse, fuse_triv and every window version are computed here.

Sealed chromosomes (results/sealed.tsv) are refused without --unseal; an unsealing is logged.

usage:
  python bench/hard/evaluate.py --labels DIR/labels_panel --published DIR/labels --genomes results/dev_genomes.tsv \\
      --merge Shigella=Escherichia --scores DIR/dev_composition:gc3,codon,kmer,short DIR/dev_wide DIR/dev_views/genus \\
      --scores-v2 DIR/dev_views/family --choose-fusion --out results/dev
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402

DNA = ("gc3", "codon", "kmer")
TRIVIAL = ("short", "nohit", "n_rows", "absent_near", "best_id", "sim_anom", "copies", "viral", "random")
LINEAR = ("ai_all", "ai_narrow", "ai_gene", "wide_rule_ai")
BOTH = ("fuse", "fuse_triv")
LABEL_SETS = ("gated", "pub", "absent", "deep")
P1 = [("codon", ("wide",)), ("fuse", ("codon",)), ("wide", LINEAR)]
P2 = [("codon_nb", ("wide_nb",)), ("fuse_nb", ("codon_nb",))]


def base_name(name):
    return name[:-3] if name.endswith("_nb") else name


def input_class(name):
    b = base_name(name)
    return "both" if b in BOTH else "dna" if b in DNA or b in ("short", "random") else "similarity"


def read_scores(spec, acc):
    d, _, cols = spec.partition(":")
    path = os.path.join(d, "scores_%s.tsv" % acc)
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    want = cols.split(",") if cols else [c for c in rows[0] if c not in ("gene_id", "full")]
    out = {}
    for c in want:
        m = {}
        for r in rows:
            gid = r["gene_id"]
            if gid.startswith(acc + "__"):
                gid = gid[len(acc) + 2:]
            try:
                v = float(r[c])
            except (ValueError, TypeError, KeyError):
                continue
            if v == v:
                m[gid] = v
        out[c] = m
    return out


def chromosome(acc, labels, published):
    rows = list(csv.DictReader(open(os.path.join(labels, acc + ".tsv")), delimiter="\t"))
    rows.sort(key=lambda r: (r["contig"], int(r["start"]), int(r["end"])))
    pub = {r["gene_id"]: r for r in csv.DictReader(open(os.path.join(published, acc + ".tsv")), delimiter="\t")} if published else {}
    col = lambda k, d="": np.array([r.get(k, d) or d for r in rows], dtype=object)
    f = np.array([float(r["f"]) if r.get("f") not in (None, "") else np.nan for r in rows])
    flags = [set((r.get("flags") or "").split(",")) - {""} for r in rows]
    ch = dict(acc=acc, gene=[r["gene_id"] for r in rows], contig=col("contig"), cls=col("class", "unresolved"), unit=col("region"),
              length=np.array([(int(r["end"]) - int(r["start"]) + 1) // 3 for r in rows]), f=f,
              absent=np.array(["absent" in x for x in flags]), deep=np.array(["deep" in x for x in flags]), fast=np.array(["fast" in x for x in flags]),
              single=col("copies", "single") == "single", n_other=int(rows[0].get("n_other") or 0) if rows else 0,
              pub=np.array([pub.get(r["gene_id"], {}).get("label", r.get("published", "none")) for r in rows], dtype=object),
              pub_unit=np.array([pub.get(r["gene_id"], {}).get("region", "") for r in rows], dtype=object))
    if len(set(ch["contig"].tolist())) > 1:
        sys.exit("%s: more than one contig in the gene table" % acc)
    ch["closed"] = bool(rows) and str(rows[0].get("closed", "1")) == "1"
    return ch


def label_set(ch, name):
    """(positive genes, negative genes, unit id per gene for positives, unit id per gene for negatives)."""
    isl, nat = ch["cls"] == "island", ch["cls"] == "native"
    unit = ch["unit"]
    is_block = np.array([str(u).startswith("NB") for u in unit])
    pu = np.where(isl & ~is_block, unit, "")
    nu = np.where(nat & is_block, unit, "")
    if name == "gated":
        return isl, nat, pu, nu
    if name == "pub":                                      # a published region is a unit when at least 3 genes carry its label
        p, n = ch["pub"] == "island", ch["pub"] == "backbone"
        ids, counts = np.unique(ch["pub_unit"][p | n], return_counts=True)
        big = np.isin(ch["pub_unit"], ids[counts >= 3])
        return p, n, np.where(p & big, ch["pub_unit"], ""), np.where(n & big, ch["pub_unit"], "")
    if name == "absent":                                   # S-a: positives absent from every other species; regions at least half absent
        keep = [u for u in set(pu.tolist()) - {""} if np.mean(ch["absent"][pu == u]) >= 0.5]
        return isl & ch["absent"], nat, np.where(np.isin(pu, keep), pu, ""), nu
    if name == "deep":                                     # S-f: thresholds from deep natives; blocks whose natives are all deep
        keep = [u for u in set(nu.tolist()) - {""} if ch["deep"][nu == u].all()]
        return isl, nat & ch["deep"], pu, np.where(np.isin(nu, keep), nu, "")
    raise ValueError(name)


def eligibility(ch):
    isl, nat = ch["cls"] == "island", ch["cls"] == "native"
    bb = (ch["pub"] == "backbone") & ch["single"]
    pi = ch["pub"] == "island"
    with np.errstate(invalid="ignore"):
        pc = float(np.mean(ch["f"][bb] >= 0.8)) if bb.any() else float("nan")
        c = float(np.mean(ch["f"][pi] >= 0.8)) if pi.any() else float("nan")
    units = set(ch["unit"].tolist()) - {""}
    blocks = sum(1 for u in units if str(u).startswith("NB"))
    regions = len(units) - blocks
    panel_ok = ch["n_other"] >= 3 and pc == pc and pc >= 0.7 and (c != c or c <= 0.2)
    n_pb, n_pi = int((ch["pub"] == "backbone").sum()), int(pi.sum())
    pub_reg = {}
    for k in ("island", "backbone"):
        _, counts = np.unique(ch["pub_unit"][(ch["pub"] == k) & (ch["pub_unit"] != "")], return_counts=True)
        pub_reg[k] = int((counts >= 3).sum())
    return dict(natives=int(nat.sum()), island_genes=int(isl.sum()), island_regions=regions, native_blocks=blocks, other_species=ch["n_other"],
                backbone_positional_core=None if pc != pc else round(pc, 4), conflict_share=None if c != c else round(c, 4),
                deep_natives=int((nat & ch["deep"]).sum()), published_island=n_pi, published_backbone=n_pb,
                p1=bool(panel_ok and nat.sum() >= 500 and isl.sum() >= 40), p2=bool(panel_ok and regions >= 5 and blocks >= 50),
                pub_gene=bool(n_pb >= 300 and n_pi >= 50), pub_block=bool(pub_reg["backbone"] >= 20 and pub_reg["island"] >= 5),
                count_flagged=bool(n_pi > n_pb))


def unit_index(unit_of_gene, mask):
    """Padded matrix of gene positions per unit (rows), -1 = padding."""
    ids = sorted(set(unit_of_gene[mask].tolist()) - {""})
    rows = [np.flatnonzero(mask & (unit_of_gene == u)) for u in ids]
    idx = np.full((len(rows), max((len(r) for r in rows), default=1)), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        idx[i, :len(r)] = r
    return idx


def unit_scores(w, scored, idx, shift=0, need=3):
    """Per unit: the median of the window values of its genes; -inf (unscored) when fewer than `need` of its genes are
    scored by the method itself. w: window value per gene (nan = none); scored: the gene has an own score."""
    if not len(idx):
        return np.zeros(0)
    pos = (idx - shift) % len(w)
    v = np.where(idx >= 0, w[pos], np.nan)
    own = np.where(idx >= 0, scored[pos], False).sum(1)
    ok = (own >= need) & np.isfinite(v).any(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(ok[:, None], v, np.nan), axis=1)
    return np.where(ok, med, -np.inf)


def found(pos, neg, allowed):
    """Share of positive units strictly above the (allowed + 1)-th highest scored negative unit. An unscored positive
    unit is missed; with fewer than allowed + 1 scored negative units every scored positive unit is found."""
    if not len(pos):
        return float("nan")
    neg = neg[np.isfinite(neg)]
    if len(neg) <= allowed:
        return float(np.mean(np.isfinite(pos)))
    return float(np.mean(pos > np.sort(neg)[::-1][allowed]))


def gene_metrics(x, pos, neg):
    """x: one value per gene, nan = unscored (ranks last, never called)."""
    v = np.where(np.isfinite(x), x, -1.0)
    p, n = v[pos], v[neg]
    sp, sn = np.isfinite(x[pos]), np.isfinite(x[neg])
    out = dict(r1=M.recall_never(p, n, sp, sn, 0.01), mr5=M.mean_recall_never(p, n, sp, sn, 0.05), auroc=M.auroc(p, n), scored_pos=float(sp.mean()),
               scored_neg=float(sn.mean()))
    if len(n) >= 1000:
        out["r03"] = M.recall_never(p, n, sp, sn, 0.003)
    return out, M.threshold_at(n, 0.01), v


def evaluate_chromosome(ch, scores, sets, shifts, rng_seed=1):
    """{label set: {method: metrics}} for one chromosome and one regime. scores: {method: raw values, nan = unscored}."""
    n = len(ch["gene"])
    q = {k: M.quantile(v) for k, v in scores.items()}
    w = {k + "_nb": M.window_scored(x, ch["contig"], 5, ch["closed"]) for k, x in q.items()}
    low = {k: np.where(np.isfinite(v), v, (np.nanmin(v) if np.isfinite(v).any() else 0.0) - 1.0) for k, v in scores.items()}
    sw = {k + "_nb": M.rank_window(v, ch["contig"], 5, ch["closed"]) for k, v in low.items()}        # S-w: unscored genes at the lowest rank
    rng = np.random.default_rng(rng_seed)
    offsets = rng.integers(1, n, shifts) if shifts and n > 1 else np.zeros(0, dtype=np.int64)
    out = {}
    for s in sets:
        pos, neg, pu, nu = label_set(ch, s)
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        has_units = (pu != "").any() and (nu != "").any()
        pi, ni = (unit_index(pu, pos), unit_index(nu, neg)) if has_units else (np.zeros((0, 1), dtype=np.int64),) * 2
        n_reg, n_blk = len(pi), len(ni)
        allowed = 1 if s == "pub" else max(1, int(np.floor(0.02 * n_blk)))
        pidx, nidx = np.flatnonzero(pos), np.flatnonzero(neg)
        res = {}
        for k, x in q.items():
            m, thr, v = gene_metrics(x, pos, neg)
            called = v > thr
            m["calls"] = {c: int((called & (ch["cls"] == c)).sum()) for c in ("island", "native", "conflict")}
            m["calls"]["unresolved"] = int(called.sum()) - sum(m["calls"].values())
            if n_reg:
                m["r1_region_weighted"] = float(np.mean([np.mean(called[pu == u]) for u in set(pu.tolist()) - {""}]))
            res[k] = m
        for k, x in w.items():
            m, _, v = gene_metrics(x, pos, neg)
            if len(offsets):
                fin = np.isfinite(x)
                m["r1_shifted"] = float(np.mean([M.recall_never(v[(pidx - o) % n], v[(nidx - o) % n], fin[(pidx - o) % n], fin[(nidx - o) % n], 0.01) for o in offsets[:100]]))
            if has_units:
                own = np.isfinite(q[k[:-3]])
                everyone = np.ones(n, dtype=bool)
                a, b = unit_scores(x, own, pi), unit_scores(x, own, ni)
                low1 = lambda z: np.where(np.isfinite(z), z, -1.0)
                m.update(lr2=found(a, b, allowed), lr1=found(a, b, 1 if s == "pub" else int(np.floor(0.01 * n_blk))), lr0=found(a, b, 0),
                         region_auroc=M.auroc(low1(a), low1(b)), regions=n_reg, blocks=n_blk, allowed=allowed,
                         regions_unscored=int((~np.isfinite(a)).sum()), blocks_unscored=int((~np.isfinite(b)).sum()),
                         lr2_sw=found(unit_scores(sw[k], everyone, pi, 0, 1), unit_scores(sw[k], everyone, ni, 0, 1), allowed))
                if len(offsets):
                    m["lr2_shifted"] = float(np.mean([found(unit_scores(x, own, pi, int(o)), unit_scores(x, own, ni, int(o)), allowed) for o in offsets]))
            res[k] = m
        res["_n"] = dict(pos=int(pos.sum()), neg=int(neg.sum()), regions=n_reg, blocks=n_blk)
        out[s] = res
    return out


def add_fusions(scores, candidate):
    """fuse and fuse_triv: codon with wide / absent_near."""
    if "codon" in scores:
        qc = M.quantile(scores["codon"])
        for name, other in (("fuse", "wide"), ("fuse_triv", "absent_near")):
            if other in scores:
                scores[name] = M.fuse(qc, M.quantile(scores[other]), candidate)
    return scores


def contrast(vals, a, bset, units):
    """Paired contrast of a against every member of bset over units. A set is an intersection-union test: the smallest
    mean difference, the smallest lower limit, the largest p."""
    best = None
    for b in bset:
        d = np.array([vals[u][a] - vals[u][b] for u in units if a in vals.get(u, {}) and b in vals.get(u, {})])
        if not len(d):
            return None
        m, lo, hi = M.t_interval(d)
        elo, ehi = M.sign_flip_interval(d)
        p = M.sign_flip_p(d)
        if len(bset) > 1 and not m > 0:                       # a member of a set that is not beaten cannot make the set significant
            p = 1.0
        r = dict(against=b, mean=m, lo=lo, hi=hi, p=p, n=len(d), wins=int((d > 0).sum()), eq_lo=elo, eq_hi=ehi)
        if best is None:
            best = r
        else:
            keep = r if r["mean"] < best["mean"] else best
            best = dict(keep, lo=min(lo, best["lo"]), p=max(r["p"], best["p"]), eq_lo=min(elo, best["eq_lo"]), eq_hi=max(ehi, best["eq_hi"]))
    return best


def excludes_zero(c):
    return c["lo"] == c["lo"] and (c["lo"] > 0 or c["hi"] < 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="panel labels (bench/hard/panel_labels.py)")
    ap.add_argument("--published", default="", help="published labels (bench/islands/labels.py), for S-pub")
    ap.add_argument("--genomes", required=True, help="table with the columns genus and accession")
    ap.add_argument("--accessions", nargs="*", default=[], help="only these chromosomes")
    ap.add_argument("--merge", nargs="*", default=[], help="Genus=Unit: genera that form one unit")
    ap.add_argument("--scores", nargs="+", required=True, help="DIR[:col,col] under the ranked regime (V1)")
    ap.add_argument("--scores-v2", nargs="*", default=[], help="DIR[:col,col] of the similarity scores with the family hidden; the other scores come from --scores")
    ap.add_argument("--fusion", default="max", choices=M.FUSIONS)
    ap.add_argument("--choose-fusion", action="store_true", help="development: print the P1 metric of all fusion candidates")
    ap.add_argument("--shifts", type=int, default=200, help="offsets of the label-shift null")
    ap.add_argument("--sealed", default=os.path.join(HERE, "results", "sealed.tsv")); ap.add_argument("--unseal", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge = dict(x.split("=", 1) for x in a.merge)
    genomes = [g for g in csv.DictReader(open(a.genomes), delimiter="\t") if not a.accessions or g["accession"] in a.accessions]
    genomes = [g for g in genomes if os.path.exists(os.path.join(a.labels, g["accession"] + ".tsv"))]
    sealed = {r["accession"]: r["tier"] for r in csv.DictReader(open(a.sealed), delimiter="\t")} if os.path.exists(a.sealed) else {}
    locked = sorted(g["accession"] for g in genomes if sealed.get(g["accession"], "open") not in ("open", a.unseal[:1]))
    if locked:
        sys.exit("sealed chromosomes without the matching --unseal: " + " ".join(locked))
    per, elig, unit_of, fusion_table = {}, {}, {}, {}
    for g in genomes:
        acc = g["accession"]
        ch = chromosome(acc, a.labels, a.published)
        unit_of[acc] = merge.get(g["genus"], g["genus"])
        elig[acc] = eligibility(ch)
        raw = {}
        for spec in a.scores:
            for k, m in read_scores(spec, acc).items():
                if k in raw:
                    sys.exit("score %s comes from two folders" % k)
                raw[k] = np.array([m.get(x, np.nan) for x in ch["gene"]])
        if not raw:
            continue
        raw2, has_v2 = dict(raw), False
        for spec in a.scores_v2:
            for k, m in read_scores(spec, acc).items():
                raw2[k] = np.array([m.get(x, np.nan) for x in ch["gene"]])
                has_v2 = True
        if a.choose_fusion and "codon" in raw and "wide" in raw:
            for cand in M.FUSIONS:
                r = evaluate_chromosome(ch, {"fuse": add_fusions(dict(raw), cand)["fuse"]}, ("gated",), 0)
                if "gated" in r:
                    fusion_table.setdefault(cand, {})[acc] = r["gated"]["fuse"]["r1"]
        per[acc] = dict(v1=evaluate_chromosome(ch, add_fusions(raw, a.fusion), LABEL_SETS if a.published else ("gated", "absent", "deep"), a.shifts))
        if has_v2:
            per[acc]["v2"] = evaluate_chromosome(ch, add_fusions(raw2, a.fusion), ("gated",), 0)
    units = sorted(set(unit_of[x] for x in per))

    def unit_values(regime, lset, metric, flag, extra=None):
        """{unit: {method: mean of the metric over the unit's chromosomes that pass the eligibility flag}}."""
        out = {}
        for u in units:
            accs = [x for x in per if unit_of[x] == u and elig[x][flag] and lset in per[x].get(regime, {}) and (extra is None or extra(x))]
            if not accs:
                continue
            names = set.intersection(*[set(per[x][regime][lset]) - {"_n"} for x in accs])
            out[u] = {k: float(np.mean([per[x][regime][lset][k][metric] for x in accs])) for k in names if all(metric in per[x][regime][lset][k] for x in accs)}
        return out

    endpoints = {"P1": ("r1", "p1", P1, "pub_gene"), "P2": ("lr2", "p2", P2, "pub_block")}
    report = {}
    for ep, (metric, flag, contrasts, pub_flag) in endpoints.items():
        main_vals = unit_values("v1", "gated", metric, flag)
        eligible = sorted(main_vals)
        size = lambda x, lset, key: per[x]["v1"].get(lset, {}).get("_n", {}).get(key, 0)
        analyses = dict(S_pub=unit_values("v1", "pub", metric, pub_flag),
                        S_a=unit_values("v1", "absent", metric, flag, (lambda x: size(x, "absent", "pos") >= 20) if ep == "P1" else (lambda x: size(x, "absent", "regions") >= 3)),
                        S_f=unit_values("v1", "deep", metric, flag, (lambda x: elig[x]["deep_natives"] >= 300) if ep == "P1" else (lambda x: size(x, "deep", "blocks") >= 20)),
                        V2=unit_values("v2", "gated", metric, flag),
                        S_b={u: v for u, v in main_vals.items() if not any(elig[x]["count_flagged"] for x in per if unit_of[x] == u)})
        rows = []
        for A, Bset in contrasts:
            c = contrast(main_vals, A, Bset, eligible)
            if c is None:
                continue
            row = dict(a=A, b=list(Bset), **c)
            row["analyses"] = {}
            for name, vals in analyses.items():
                cc = contrast(vals, A, Bset, sorted(vals))
                row["analyses"][name] = None if cc is None else dict(mean=cc["mean"], lo=cc["lo"], hi=cc["hi"], n=cc["n"])
            suffix = "_nb" if A.endswith("_nb") else ""
            triv = [t + suffix for t in TRIVIAL] + (["fuse_triv" + suffix] if input_class(A) == "both" else [])
            row["trivial"] = {}
            for t in triv:
                c2 = contrast(main_vals, A, (t,), eligible)
                row["trivial"][t] = None if c2 is None else dict(mean=c2["mean"], lo=c2["lo"])
            rows.append(row)
        for row, p in zip(rows, M.holm(np.array([r["p"] for r in rows])) if rows else []):
            row["p_holm"] = float(p)
            sign = np.sign(row["mean"])
            is_set = len(row["b"]) > 1
            classes_a, classes_b = input_class(row["a"]), {input_class(b) for b in row["b"]}
            winner_class = {classes_a} if sign > 0 else classes_b
            same_side = lambda s_: s_["lo"] == s_["lo"] and ((s_["lo"] > 0 and sign > 0) or (s_["hi"] < 0 and sign < 0))
            first, later, notes = [], [], []
            if not excludes_zero(row) or (is_set and not row["lo"] > 0):
                first.append("interval includes 0")
            if abs(row["mean"]) < 0.05:
                first.append("difference under 0.05")
            if not row["p_holm"] < 0.05:
                first.append("Holm p %.3f" % row["p_holm"])
            if is_set and sign <= 0:
                first.append("the set is not beaten")
            for name in ("S_pub", "S_a", "S_b"):
                s_ = row["analyses"].get(name)
                label = name.replace("_", "-")
                if s_ is None:
                    later.append(label + " has no genus")
                elif np.sign(s_["mean"]) != sign or s_["mean"] == 0:
                    later.append(label + " not the same sign")
                elif s_["n"] < 5:
                    notes.append("%s on %d genera" % (label, s_["n"]))
            if "similarity" in {classes_a} | classes_b or "both" in {classes_a} | classes_b:
                s_ = row["analyses"].get("V2")
                if s_ is None:
                    if winner_class & {"similarity", "both"}:
                        later.append("V2 not evaluable")
                    else:
                        notes.append("V2 not evaluable")
                elif np.sign(s_["mean"]) != sign or (s_["n"] >= 7 and not same_side(s_)):
                    later.append("V2 does not agree")
                elif s_["n"] < 7:
                    notes.append("V2 not tested (sign only, %d genera)" % s_["n"])
            if classes_b != {classes_a}:
                s_ = row["analyses"].get("S_f")
                if s_ is None:
                    if winner_class == {"dna"}:
                        later.append("S-f not evaluable")
                    else:
                        notes.append("S-f not evaluable")
                elif np.sign(s_["mean"]) != sign or (s_["n"] >= 7 and not same_side(s_)):
                    later.append("S-f does not agree")
                elif s_["n"] < 7:
                    notes.append("S-f not tested (sign only, %d genera)" % s_["n"])
            lost = [t for t, v in row["trivial"].items() if v is not None and not (v["mean"] > 0 and v["lo"] > 0)]
            winner = (row["a"], "+".join(row["b"])) if sign > 0 else ("+".join(row["b"]), row["a"])
            if first:
                row["verdict"] = "equivalent at this resolution" if (row["eq_lo"] > -0.05 and row["eq_hi"] < 0.05) else "unresolved"
            elif later:
                row["verdict"] = "unresolved"
            elif sign > 0 and lost:
                row["verdict"] = "%s above %s, but %s" % (winner[0], winner[1], "fusion not shown to add to composition plus a trivial similarity proxy"
                                                         if lost == ["fuse_triv" + ("_nb" if row["a"].endswith("_nb") else "")] else "not shown to detect origin beyond a trivial proxy")
            else:
                row["verdict"] = "%s better than %s" % winner
            if sign > 0 and lost:
                later.append("not above trivial: " + ", ".join(lost))
            row["open"], row["notes"] = first + later, notes
        report[ep] = dict(units=eligible, values=main_vals, contrasts=rows, min_p=None if not eligible else 2 / 2 ** len(eligible))

    def cand_mean(t):
        v = [np.mean([x for acc, x in t.items() if unit_of[acc] == u and elig[acc]["p1"]]) for u in units if any(unit_of[acc] == u and elig[acc]["p1"] for acc in t)]
        return float(np.mean(v)) if v else None

    result = dict(endpoints=report, eligibility=elig, units=unit_of, per_chromosome=per, fusion=a.fusion,
                  fusion_candidates={c: dict(mean_over_units=cand_mean(t), per_chromosome=t) for c, t in fusion_table.items()})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if o != o or o in (float("inf"), float("-inf")) else round(float(o), 5)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    json.dump(clean(result), open(a.out + ".json", "w"), indent=1)
    if a.unseal:
        h = hashlib.sha256()
        for f in sorted(os.path.join(a.labels, g["accession"] + ".tsv") for g in genomes) + [os.path.abspath(__file__), os.path.join(HERE, "metrics.py")]:
            h.update(open(f, "rb").read())
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
        with open(os.path.join(HERE, "results", "unseal.log"), "a") as fo:
            fo.write(json.dumps(dict(unseal=a.unseal, commit=commit, labels_and_evaluator_sha256=h.hexdigest(),
                                     chromosomes=[g["accession"] for g in genomes], out=os.path.basename(a.out))) + "\n")
    f3 = lambda x: "" if x is None or x != x else "%.3f" % x
    L = ["Chromosomes %d, units %d. Fusion candidate: %s." % (len(per), len(units), a.fusion), ""]
    for ep, r in report.items():
        title = "R@1% at gene level, own score" if ep == "P1" else "island regions found (LR@2%) at block level, window of 11 genes"
        names = sorted({k for v in r["values"].values() for k in v if k.endswith("_nb") == (ep == "P2")},
                       key=lambda k: -np.mean([v[k] for v in r["values"].values() if k in v]))
        L += ["## %s: %s" % (ep, title), "", "Eligible units: %d (%s)." % (len(r["units"]), ", ".join(r["units"])), "",
              "| score | input | mean over units | " + " | ".join(r["units"]) + " |", "|---|---|---|" + "---|" * len(r["units"])]
        for k in names:
            vals = [r["values"][u].get(k) for u in r["units"]]
            kind = "trivial" if base_name(k) in TRIVIAL + ("fuse_triv",) else input_class(k)
            L.append("| %s | %s | %s | %s |" % (k, kind, f3(np.mean([v for v in vals if v is not None])), " | ".join(f3(v) for v in vals)))
        if r["contrasts"]:
            L += ["", "| contrast | difference (95% t-interval) | units won | p (Holm) | S-pub | S-a | S-b | S-f | V2 | verdict | open conditions |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in r["contrasts"]:
                an = lambda k: "" if c["analyses"].get(k) is None else "%+.3f (n %d)" % (c["analyses"][k]["mean"], c["analyses"][k]["n"])
                L.append("| %s - %s | %+.3f (%+.3f to %+.3f) | %d of %d | %.4f (%.4f) | %s | %s | %s | %s | %s | %s | %s |" % (
                    c["a"], "min of " + ", ".join(c["b"]) if len(c["b"]) > 1 else c["b"][0], c["mean"], c["lo"], c["hi"], c["wins"], c["n"], c["p"], c["p_holm"],
                    an("S_pub"), an("S_a"), an("S_b"), an("S_f"), an("V2"), c["verdict"], "; ".join(c["open"] + c.get("notes", []))))
        L.append("")
    if fusion_table:
        L += ["Fusion candidates (P1 metric, mean over eligible units): " + ", ".join(
            "%s %s" % (c, f3(result["fusion_candidates"][c]["mean_over_units"])) for c in M.FUSIONS if c in result["fusion_candidates"]) + ".", ""]
    L += ["## Eligibility (label files only)", "",
          "| chromosome | unit | natives | island genes | island regions | native blocks | other species | backbone in positional core | conflict share | P1 | P2 | S-pub gene / block |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for acc, e in elig.items():
        L.append("| %s | %s | %d | %d | %d | %d | %d | %s | %s | %s | %s | %s / %s |" % (
            acc, unit_of[acc], e["natives"], e["island_genes"], e["island_regions"], e["native_blocks"], e["other_species"], f3(e["backbone_positional_core"]),
            f3(e["conflict_share"]), "yes" if e["p1"] else "no", "yes" if e["p2"] else "no", "yes" if e["pub_gene"] else "no", "yes" if e["pub_block"] else "no"))
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
