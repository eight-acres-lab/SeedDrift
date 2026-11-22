"""The readouts of package W18 on ruler tables and their AUCs (protocol w18-v0), from the per-cell bits of the passes of
w18_passes.py. numpy only: no torch, no ruler generator. The readouts' one implementation; W11 reads through it.

Every score is oriented so that higher means more foreign. Per table:
  source_shift   1/2 sum_t |pT(t) - pC(t)| over the tiers t (of the rows' distance to A) present, where pT is the share of
                 tier t in the positive costs dT(t) = mean bits of the target's scattered cells with tier t removed minus
                 with every row, and pC the same over the context cells (neighbours and relation blocks); 0 with one tier;
                 undefined (nan) when either block relies on no tier (sum of positive costs below UNDEFINED bits per codon)
  abstain        bits of the target's scattered cells without any other row minus with every row below ABSTAIN (a table
                 without other rows abstains): reported, not scored
  gain           the two-block gain: [sum over the first half (W - H1) + sum over the second half (W - H2)] / (2 cells)
  composition    -log2 p(codon | amino acid) of the arm without relatives under pass N, over the cells kept (truth not
                 OTHER, not the gene's last codon, an amino acid of two codons or more), per kept cell; composition_rel
                 divides each cell by log2 of its amino acid's codons first
Controls (no model): best_similarity (the highest amino-acid identity on the target block among the other rows minus the
highest among the rows of the nearest tier present; empty sets give 0; a row counts with 10 placed columns or more),
rows_on_target, covered_share, host_codon and host_codon_rel (the composition readout with p(codon | amino acid) from A's
codon counts over its other genes, plus 0.5 per codon). The untrained model and the presence passes go through the same
readouts.

AUC (auc_rows): a table is the unit; AUC = P(implant > control) + 1/2 P(tie) over pairs of the same group of A and the
same stratum, per group, then the plain mean over the readable groups of a block; intervals by a bootstrap over A's
genera stratified by group (weights on genera, so a replicate is a quadratic form on the genus x genus pair counts), the
same replicates for every score, so differences are paired.
"""
from __future__ import annotations

import warnings

import numpy as np

from seeddrift.train.rel import api
from seeddrift.train.rel.evalh import score as SC
from seeddrift.train.rel.readouts import w18_passes as P

VERSION = 1
UNDEFINED = 0.02
ABSTAIN = 0.05
MIN_COLUMNS = 10
PSEUDO = 0.5
ENTROPY_BINS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf)
UNCALIBRATED = 0.1
IMPLANT, ORIGINAL, OWN = 0, 1, 2
COMPARISONS = (("own", OWN), ("original", ORIGINAL))
BOOT, BOOT_SEED = 2000, 20261124
MIN_GROUP, MIN_GROUPS, MIN_IMPLANTS, MIN_GENERA = 5, 3, 30, 8
NEAREST_BINS = ((0.0, 0.3, "nearest<0.3"), (0.3, 0.5, "nearest=0.3-0.5"), (0.5, 0.7, "nearest=0.5-0.7"), (0.7, 1.01, "nearest>=0.7"))
FILL_BINS = ((0, 0, "rows=0"), (1, 1, "rows=1"), (2, 3, "rows=2-3"), (4, 7, "rows=4-7"), (8, 99, "rows=8-15"))
T0_, T1_ = P.T0_, P.T1_


def _classes():
    cls = SC.amino_acid_classes()                       # 1..20 amino acids, 21 stop, 22 other; the genetic code, reporting only
    deg = np.bincount(cls[1:65], minlength=23)
    deg[22] = 1
    return cls, deg


def pass_sums(cells, n_tables):
    """{pass name: (sum of bits [N], cells [N])} and the half sums of W ({'W1','W2'}) over the scored cells."""
    out = {}
    pid, tab, bits = cells["pass_id"].astype(np.int64), cells["table"].astype(np.int64), cells["bits"].astype(np.float64)
    names = [str(x) for x in cells["pass_names"]]
    for k in np.unique(pid):
        m = pid == k
        s, c = np.zeros(n_tables), np.zeros(n_tables)
        np.add.at(s, tab[m], bits[m])
        np.add.at(c, tab[m], 1)
        out[names[k]] = (s, c)
    return out


def _mean(sums, name, n):
    s, c = sums.get(name, (np.zeros(n), np.zeros(n)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _half_sums(cells, n_tables, target_len, name, half):
    """Sum of bits and cells of pass `name` over the first (half 0) or second (half 1) half of the target."""
    names = [str(x) for x in cells["pass_names"]]
    if name not in names:
        return np.zeros(n_tables), np.zeros(n_tables)
    m = cells["pass_id"].astype(np.int64) == names.index(name)
    tab = cells["table"][m].astype(np.int64)
    off = cells["slot"][m].astype(np.int64) - T0_
    sec = off >= (np.asarray(target_len, dtype=np.int64)[tab] // 2)
    keep = sec if half else ~sec
    s, c = np.zeros(n_tables), np.zeros(n_tables)
    np.add.at(s, tab[keep], cells["bits"][m][keep].astype(np.float64))
    np.add.at(c, tab[keep], 1)
    return s, c


def readouts(cells, meta, prefix=""):
    """{score name: [N] per table} of the source shift, its undefined-at-top version, abstention and the gain, from the
    cells of one arm with relatives (prefix "P:" reads the presence passes)."""
    n = len(meta["triple"])
    sums = pass_sums(cells, n)
    tiers = meta["row_tier"]
    t_all, c_all = _mean(sums, prefix + "T", n), _mean(sums, prefix + "C", n)
    removed = {(f, k): _mean(sums, "%s%s-t%d" % (prefix, f, k), n) for f in ("T", "C") for k in range(P.MAX_TIERS)}
    shift = np.full(n, np.nan)
    for t in range(n):
        present = sorted({int(x) for x in tiers[t][1:] if x >= 0})
        if not np.isfinite(t_all[t]) or not np.isfinite(c_all[t]):
            continue
        dT = np.array([removed[("T", k)][t] - t_all[t] for k in present])
        dC = np.array([removed[("C", k)][t] - c_all[t] for k in present])
        if len(present) == 0 or not (np.isfinite(dT).all() and np.isfinite(dC).all()):
            continue
        pT, pC = np.maximum(dT, 0), np.maximum(dC, 0)
        if pT.sum() < UNDEFINED or pC.sum() < UNDEFINED:
            continue
        shift[t] = 0.0 if len(present) == 1 else 0.5 * np.abs(pT / pT.sum() - pC / pC.sum()).sum()
    w1, c1 = _half_sums(cells, n, meta["target_len"], prefix + "W", 0)
    w2, c2 = _half_sums(cells, n, meta["target_len"], prefix + "W", 1)
    h1, d1 = _half_sums(cells, n, meta["target_len"], prefix + "H1", 0)
    h2, d2 = _half_sums(cells, n, meta["target_len"], prefix + "H2", 1)
    ok = (c1 > 0) & (c2 > 0) & (c1 == d1) & (c2 == d2)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.where(ok, ((w1 - h1) + (w2 - h2)) / (2 * np.maximum(c1 + c2, 1)), np.nan)
    out = {"source_shift": shift, "source_shift_top": np.where(np.isnan(shift), np.inf, shift), "gain": gain}
    if not prefix:
        t0 = _mean(sums, "T0", n)
        out["abstain"] = np.where(np.isfinite(t0), (t0 - t_all) < ABSTAIN, True).astype(np.float64)
        out["abstain"][~np.isfinite(t_all)] = np.nan
    return out


def kept_codons(truth, offset, target_len, target_ends, cls, deg):
    """Cells the composition readouts keep: truth not OTHER, not the gene's last codon, an amino acid of two codons or more."""
    truth = np.asarray(truth, dtype=np.int64)
    last = np.asarray(target_ends, dtype=bool) & (np.asarray(offset) == np.asarray(target_len) - 1)
    return (truth >= 1) & (truth <= 64) & ~last & (deg[cls[np.clip(truth, 0, 65)]] >= 2)


def composition(cells, meta):
    """{composition, composition_rel: [N]} from the cells of pass N of an arm without relatives."""
    n = len(meta["triple"])
    cls, deg = _classes()
    names = [str(x) for x in cells["pass_names"]]
    out = {"composition": np.full(n, np.nan), "composition_rel": np.full(n, np.nan)}
    if "N" not in names:
        return out
    m = cells["pass_id"].astype(np.int64) == names.index("N")
    tab = cells["table"][m].astype(np.int64)
    truth = cells["truth"][m].astype(np.int64)
    off = cells["slot"][m].astype(np.int64) - T0_
    keep = kept_codons(truth, off, meta["target_len"][tab], meta["target_ends"][tab], cls, deg)
    c = (cells["bits"][m] - cells["aa_bits"][m]).astype(np.float64)
    rel = c / np.log2(np.maximum(deg[cls[np.clip(truth, 0, 65)]], 2))
    for name, v in (("composition", c), ("composition_rel", rel)):
        s, k = np.zeros(n), np.zeros(n)
        np.add.at(s, tab[keep], v[keep])
        np.add.at(k, tab[keep], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(k > 0, s / np.maximum(k, 1), np.nan)
    return out


def calibration(cells, meta, names=("W", "H1", "H2")):
    """Reliability by bits per variant: {variant: [{bin, cells, entropy, bits}], plus the calibration error}."""
    pn = [str(x) for x in cells["pass_names"]]
    ids = [pn.index(x) for x in names if x in pn]
    m = np.isin(cells["pass_id"].astype(np.int64), ids)
    var = meta["variant"][cells["table"][m].astype(np.int64)]
    H, b = cells["entropy"][m].astype(np.float64), cells["bits"][m].astype(np.float64)
    which = np.digitize(H, ENTROPY_BINS[1:-1])
    out = {}
    for v, vname in enumerate(("implant", "original", "own")):
        rows, err, tot = [], 0.0, 0
        for k in range(len(ENTROPY_BINS) - 1):
            sel = (var == v) & (which == k)
            c = int(sel.sum())
            rows.append(dict(bin="%g-%g" % (ENTROPY_BINS[k], ENTROPY_BINS[k + 1]), cells=c, entropy=float(H[sel].mean()) if c else None,
                             bits=float(b[sel].mean()) if c else None))
            if c:
                err += c * abs(float(b[sel].mean()) - float(H[sel].mean()))
                tot += c
        out[vname] = dict(bins=rows, error=err / tot if tot else None, cells=tot)
    e = out["own"]["error"]
    out["uncalibrated"] = bool(e is not None and e > UNCALIBRATED)
    return out


def controls(tables, meta):
    """{best_similarity, rows_on_target, covered_share, host_codon, host_codon_rel: [N]} (tables: tok, n, col, kind of every
    table [N, R, ...])."""
    from seeddrift.train import source_model as S            # the genetic code, for the classical controls only
    code = np.asarray(S.CODE)
    cls, deg = _classes()
    tok, n, col, kind = (np.asarray(tables[k]) for k in ("tok", "n", "col", "kind"))
    N, R = tok.shape[:2]
    tb = P.TARGET_BLOCK
    best = np.zeros(N)
    for t in range(N):
        own = tok[t, 0, T0_:T1_].astype(np.int64)
        tiers = meta["row_tier"][t]
        ids = {}
        for q in range(1, R):
            if kind[t, q, tb] != api.KIND_PRESENT:
                continue
            c = col[t, q, T0_:T0_ + int(n[t, q, tb])].astype(np.int64)
            k = tok[t, q, T0_:T0_ + int(n[t, q, tb])].astype(np.int64)
            m = c >= 0
            if m.sum() < MIN_COLUMNS:
                continue
            ids[q] = float((code[k[m]] == code[own[c[m]]]).mean())
        if ids:
            near = min(int(tiers[q]) for q in ids)
            inside = [v for q, v in ids.items() if tiers[q] == near]
            outside = [v for q, v in ids.items() if tiers[q] != near]
            best[t] = (max(outside) if outside else 0.0) - (max(inside) if inside else 0.0)
    on = meta["on_target"].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(on > 0, meta["covered"] / np.maximum(on * meta["target_len"], 1), 0.0)
    host, host_rel = np.full(N, np.nan), np.full(N, np.nan)
    for t in range(N):
        L = int(meta["target_len"][t])
        truth = tok[t, 0, T0_:T0_ + L].astype(np.int64)
        keep = kept_codons(truth, np.arange(L), L, meta["target_ends"][t], cls, deg)
        if not keep.any():
            continue
        cnt = meta["host_counts"][t].astype(np.float64) + PSEUDO            # tokens 1..64
        tot = np.zeros(23)
        np.add.at(tot, cls[1:65], cnt)
        c = -np.log2(cnt[truth[keep] - 1] / tot[cls[truth[keep]]])
        host[t] = c.mean()
        host_rel[t] = (c / np.log2(deg[cls[truth[keep]]])).mean()
    return {"best_similarity": best, "rows_on_target": on, "covered_share": cov, "host_codon": host, "host_codon_rel": host_rel}


# ---------------------------------------------------------------------------------------------------------------------
def _num(x):
    return float(x) if x is not None and np.isfinite(x) else None


def auc_matrix(sp, sn):
    """(n_pos, n_neg): 1 where the implant scores higher, 1/2 at a tie."""
    sp, sn = np.asarray(sp, dtype=np.float64)[:, None], np.asarray(sn, dtype=np.float64)[None, :]
    return (sp > sn) + 0.5 * (sp == sn)


def auc(sp, sn):
    sp, sn = np.asarray(sp, dtype=np.float64), np.asarray(sn, dtype=np.float64)
    if not len(sp) or not len(sn):
        return np.nan
    return float(auc_matrix(sp, sn).mean())


class Bootstrap:
    """Genus weights of the replicates, per group of A: weights[group] (B, genera of the group), multinomial."""

    def __init__(self, group, genus, reps=BOOT, seed=BOOT_SEED):
        self.group, self.genus = np.asarray(group), np.asarray(genus)
        self.reps = int(reps)
        self.genera, self.weights, self.index = {}, {}, {}
        for k, g in enumerate(sorted(set(self.group.tolist()))):
            gen = sorted(set(self.genus[self.group == g].tolist()))
            self.genera[g] = gen
            rng = np.random.default_rng([int(seed), k])
            self.weights[g] = rng.multinomial(len(gen), np.full(len(gen), 1.0 / len(gen)), size=self.reps).astype(np.float64)
            self.index[g] = {x: i for i, x in enumerate(gen)}

    def group_auc(self, g, sp, gp, sn, gn):
        """(AUC, replicates [B]) of one group: sp, sn the scores, gp, gn the genera of the implant and control tables."""
        if not len(sp) or not len(sn):
            return np.nan, np.full(self.reps, np.nan)
        K = auc_matrix(sp, sn)
        G = len(self.genera[g])
        ip = np.array([self.index[g][x] for x in gp])
        jn = np.array([self.index[g][x] for x in gn])
        M = np.zeros((G, G))
        np.add.at(M, (ip[:, None].repeat(len(jn), 1), jn[None, :].repeat(len(ip), 0)), K)
        Pc, Nc = np.bincount(ip, minlength=G).astype(np.float64), np.bincount(jn, minlength=G).astype(np.float64)
        W = self.weights[g]
        num = np.einsum("bg,gh,bh->b", W, M, W)
        den = (W @ Pc) * (W @ Nc)
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        return float(K.mean()), reps


def stratum_masks(meta, tables_identity, donor_block_same):
    """{(kind, name): bool [N]} of the strata of the protocol; the triple-level ones mark the three tables of a triple."""
    n = len(meta["triple"])
    out = {("all", "all"): np.ones(n, dtype=bool)}
    for t in sorted(set(meta["donor_tier"].tolist())):
        out[("donor_tier", "tier=%d" % t)] = meta["donor_tier"] == t
    near = SC.nearest_identity(tables_identity)
    for lo, hi, name in NEAREST_BINS:
        out[("nearest", name)] = (near >= lo) & (near < hi)
    out[("nearest", "nearest=none")] = ~np.isfinite(near)
    for lo, hi, name in FILL_BINS:
        out[("rows", name)] = (meta["on_target"] >= lo) & (meta["on_target"] <= hi)
    out[("donor_block", "same")] = donor_block_same
    out[("donor_block", "other")] = ~donor_block_same
    return out


def auc_rows(scores, meta, groups_of_table, group_names, strata, boot, label=None, reference=None):
    """Result rows of the AUCs of `scores` ({name: [N]}): per comparison, block and stratum, the group-mean AUC with its
    interval, counts and whether it is readable; with reference ({control name: [N]} of the same tables), for every score
    and control `vs_<control>`: both AUCs on the tables where both are defined and the difference with its interval over
    the same replicates (paired by genus)."""
    rows = []
    var = meta["variant"]
    genus = meta["genus_a"]
    blocks = {}
    for k, name in enumerate(group_names):
        blocks.setdefault(SC.block_of(name), []).append(k)
    for sname, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        for cname, cv in COMPARISONS:
            for (skind, stratum), mask in strata.items():
                per_block = {}
                for g in sorted(set(groups_of_table.tolist())):
                    pos = mask & (var == IMPLANT) & (groups_of_table == g) & ~np.isnan(s)
                    neg = mask & (var == cv) & (groups_of_table == g) & ~np.isnan(s)
                    a, reps = boot.group_auc(g, s[pos], genus[pos], s[neg], genus[neg])
                    per_block.setdefault(SC.block_of(group_names[g]), []).append((g, pos.sum(), neg.sum(), a, reps, set(genus[pos].tolist())))
                for block, items in sorted(per_block.items()):
                    enter = [x for x in items if x[1] >= MIN_GROUP and x[2] >= MIN_GROUP]
                    n_pos = sum(int(x[1]) for x in enter)
                    genera = len(set().union(*[x[5] for x in enter])) if enter else 0
                    readable = len(enter) >= MIN_GROUPS and n_pos >= MIN_IMPLANTS and genera >= MIN_GENERA
                    if enter:
                        point = float(np.mean([x[3] for x in enter]))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)       # a replicate that drew no genus of a class
                            reps = np.nanmean(np.vstack([x[4] for x in enter]), axis=0)
                        lo, hi = (float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))) if np.isfinite(reps).any() else (None, None)
                    else:
                        point, reps, lo, hi = None, None, None, None
                    row = dict(score=sname, label=label or "", comparison=cname, block=block, stratum_kind=skind, stratum=stratum, groups=len(enter),
                               groups_with_tables=sum(1 for x in items if x[1] and x[2]), implants=n_pos, controls=sum(int(x[2]) for x in enter),
                               genera=genera, auc=point, lo=lo, hi=hi, readable=bool(readable))
                    for rname, r in (reference or {}).items():
                        if not enter:
                            break
                        r = np.asarray(r, dtype=np.float64)
                        both = []
                        for g, _, _, _, _, _ in enter:                 # the same tables for both, the same replicates
                            pos = mask & (var == IMPLANT) & (groups_of_table == g) & ~np.isnan(s) & ~np.isnan(r)
                            neg = mask & (var == cv) & (groups_of_table == g) & ~np.isnan(s) & ~np.isnan(r)
                            both.append((boot.group_auc(g, s[pos], genus[pos], s[neg], genus[neg]), boot.group_auc(g, r[pos], genus[pos], r[neg], genus[neg])))
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)       # a group without a table both define
                            a_s = float(np.nanmean([x[0][0] for x in both]))
                            a_r = float(np.nanmean([x[1][0] for x in both]))
                            d = np.nanmean(np.vstack([x[0][1] for x in both]), axis=0) - np.nanmean(np.vstack([x[1][1] for x in both]), axis=0)
                        fin = np.isfinite(d).any()
                        row["vs_" + rname] = dict(auc=_num(a_s), reference_auc=_num(a_r), diff=_num(a_s - a_r), lo=float(np.nanpercentile(d, 2.5)) if fin else None,
                                                  hi=float(np.nanpercentile(d, 97.5)) if fin else None)
                    rows.append(row)
    return rows


def per_group_rows(scores, meta, groups_of_table, group_names, label=None):
    """The all-stratum AUC of every group with its counts (implant against own and against original)."""
    rows = []
    var = meta["variant"]
    for sname, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        for cname, cv in COMPARISONS:
            for g in sorted(set(groups_of_table.tolist())):
                pos = (var == IMPLANT) & (groups_of_table == g) & ~np.isnan(s)
                neg = (var == cv) & (groups_of_table == g) & ~np.isnan(s)
                rows.append(dict(score=sname, label=label or "", comparison=cname, group=group_names[g], implants=int(pos.sum()), controls=int(neg.sum()),
                                 genera=len(set(meta["genus_a"][pos].tolist())), auc=auc(s[pos], s[neg]) if pos.any() and neg.any() else None))
    return rows
